"""Detector in tempo reale basato su YOLOE (open-vocabulary con segmentazione). Prova sulla webcam del PC: python3 detector.py bottle"""
import atexit
import contextlib
import os
import shutil
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
warnings.filterwarnings("ignore", message=".*torch.jit.load.*", category=FutureWarning)

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL = "yoloe-11s-seg.pt"
WINDOW = "detector"
IGNORE_IOU = 0.3
IGNORE_MISSES = 10
CROP_PAD = 0.12
CROP_FILL = (124, 124, 124)


class WebcamSource:

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"webcam {index} non disponibile")
        self._seq = 0

    def get_frame(self, after=-1):
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._seq += 1
        return self._seq, frame

    def close(self):
        self.cap.release()


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def masked_crop(frame, det):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = det["box"]
    px, py = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
    x1, y1 = max(0, round(x1 - px)), max(0, round(y1 - py))
    x2, y2 = min(w, round(x2 + px)), min(h, round(y2 + py))
    region = frame[y1:y2, x1:x2]
    mask = det.get("mask")
    if mask is not None and len(mask) >= 3 and region.size:
        full = np.zeros((h, w), np.uint8)
        cv2.fillPoly(full, [np.asarray(mask, np.int32)], 255)
        full = cv2.dilate(full, np.ones((7, 7), np.uint8))
        region = region.copy()
        region[full[y1:y2, x1:x2] == 0] = CROP_FILL
    return region


class LiveDetector:
    def __init__(self, threshold=0.5, min_conf=0.25, imgsz=640, show=True, model=MODEL,
                 overrides=None):
        from ultralytics import YOLOE

        self.threshold, self.min_conf, self.imgsz, self.show = threshold, min_conf, imgsz, show
        self.overrides = overrides or {}
        MODELS_DIR.mkdir(exist_ok=True)
        with contextlib.chdir(MODELS_DIR):
            self.model = YOLOE(MODEL if model == MODEL else model)
        self._query = None
        self._model_lock = threading.Lock()
        self._job = None
        self._idle = threading.Event()
        self._idle.set()
        self._view = None
        self._annotated = None
        self._thread = None
        self._closing = False

    def detect(self, query, frame):
        if query in self.overrides:
            return self.overrides[query](frame)
        with self._model_lock:
            if query != self._query:
                with contextlib.chdir(MODELS_DIR):
                    self.model.set_classes([query], self.model.get_text_pe([query]))
                self._query = query
            res = self.model.predict(frame, conf=self.min_conf, imgsz=self.imgsz,
                                     verbose=False)[0]
        masks = res.masks.xy if res.masks is not None else [None] * len(res.boxes)
        detections = [{"label": query, "confidence": round(float(c), 2),
                       "box": [round(float(v)) for v in xyxy], "mask": mask}
                      for xyxy, c, mask in zip(res.boxes.xyxy, res.boxes.conf, masks)]
        detections.sort(key=lambda d: -d["confidence"])
        return detections, res.plot()

    def segment(self, frame, box):
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        with self._model_lock:
            res = self.model.predict(
                frame, refer_image=frame, predictor=YOLOEVPSegPredictor,
                visual_prompts={"bboxes": np.array([box], float), "cls": np.array([0])},
                conf=self.min_conf, imgsz=self.imgsz, verbose=False)[0]
            self._query = None
        if res.masks is None:
            return None
        overlap = [_iou([float(v) for v in b], box) for b in res.boxes.xyxy]
        best = int(np.argmax(overlap))
        return res.masks.xy[best] if overlap[best] > 0.5 else None

    def prepare(self, query):
        self.detect(query, np.zeros((self.imgsz, self.imgsz, 3), np.uint8))

    def start(self, query, get_frame, out_dir, on_hit=None, ignore=(), watch_only=False):
        self.stop(keep_view=True)
        self._job = {"query": query, "get_frame": get_frame, "out_dir": Path(out_dir),
                     "on_hit": on_hit, "hit": None, "best": None, "frames": 0,
                     "ignore": [list(b) for b in ignore], "watch_only": watch_only}
        self._idle.clear()
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            atexit.register(self.close)

    def stop(self, keep_view=False):
        job, self._job = self._job, None
        if job is None:
            return {"hit": None, "best": None, "frames": 0, "ignore": []}
        self._idle.wait(5.0)
        if not keep_view:
            self._annotated = None
        return {k: job[k] for k in ("hit", "best", "frames", "ignore")}

    def view(self):
        return self._annotated

    def clear_view(self):
        self._annotated = None

    def close(self):
        self.stop()
        self._closing = True
        if self._thread is not None:
            self._thread.join(5.0)
            self._thread = None

    def _save(self, job, name, annotated, det, raw=None):
        job["out_dir"].mkdir(parents=True, exist_ok=True)
        stem = f"{name}_{datetime.now():%H%M%S}_{det['confidence']:.2f}"
        path = job["out_dir"] / f"{stem}.jpg"
        cv2.imwrite(str(path), annotated)
        out = {k: v for k, v in det.items() if k != "mask"}
        out["image"] = str(path)
        if raw is not None:
            raw_path = job["out_dir"] / f"{stem}_raw.jpg"
            cv2.imwrite(str(raw_path), raw)
            out["raw_image"] = str(raw_path)
            crop_path = job["out_dir"] / f"{stem}_crop.png"
            cv2.imwrite(str(crop_path), masked_crop(raw, det))
            out["crop_image"] = str(crop_path)
        return out

    def _drop_ignored(self, job, detections):
        kept, tracked = [], []
        for det in detections:
            if any(_iou(det["box"], box) > IGNORE_IOU for box in job["ignore"]):
                tracked.append(det["box"])
            else:
                kept.append(det)
        if job["ignore"]:
            job["misses"] = 0 if tracked else job.get("misses", 0) + 1
            if tracked:
                job["ignore"] = tracked
            elif job["misses"] > IGNORE_MISSES:
                job["ignore"] = []
        return kept

    def _loop(self):
        last_seq = -1
        while not self._closing:
            job = self._job
            if job is None or job["hit"] is not None:
                self._idle.set()
                self._show(None)
                continue
            got = job["get_frame"](last_seq)
            if got is None:
                self._show(None)
                continue
            last_seq, frame = got
            detections, annotated = self.detect(job["query"], frame)
            if self._job is job:
                self._annotated = annotated
            if job["watch_only"]:
                self._show(annotated)
                continue
            job["frames"] += 1
            detections = self._drop_ignored(job, detections)
            if detections:
                top = detections[0]
                if top["confidence"] >= self.threshold:
                    job["hit"] = self._save(job, "found", annotated, top, raw=frame)
                    if job["on_hit"]:
                        job["on_hit"]()
                elif job["best"] is None or top["confidence"] > job["best"]["confidence"]:
                    old = job["best"]
                    job["best"] = self._save(job, "best", annotated, top)
                    if old:
                        Path(old["image"]).unlink(missing_ok=True)
            self._show(annotated)
        if self._view is not None:
            cv2.destroyAllWindows()

    def _show(self, frame):
        if not self.show:
            if frame is None:
                time.sleep(0.02)
            return
        try:
            if frame is not None:
                self._view = frame
                cv2.imshow(WINDOW, frame)
            if self._view is not None:
                cv2.waitKey(1 if frame is not None else 20)
            else:
                time.sleep(0.02)
        except cv2.error:
            self.show = False


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Prova il detector con la webcam del PC")
    ap.add_argument("query", help='etichetta in inglese, es. "bottle"')
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--no-stop", action="store_true",
                    help="non fermarti al primo oggetto trovato: video fino alla fine")
    args = ap.parse_args()

    cam = WebcamSource(args.camera)
    det = LiveDetector(threshold=2.0 if args.no_stop else args.threshold,
                       show=not args.no_show)
    found = threading.Event()
    out = Path(__file__).resolve().parent / "inspect_shots" / "prova_webcam"
    try:
        det.start(args.query, cam.get_frame, out, on_hit=found.set)
        found.wait(args.seconds)
        result = det.stop()
        hit, best = result["hit"], result["best"]
        if hit:
            print(f"TROVATO '{args.query}' con confidenza {hit['confidence']} "
                  f"dopo {result['frames']} frame: mi fermo (è il comportamento "
                  "previsto, usa --no-stop per continuare a guardare).")
        else:
            print(f"Tempo scaduto dopo {result['frames']} frame. Confidenza massima: "
                  f"{best['confidence'] if best else 'nessun box'}.")
        if hit and det.show:
            time.sleep(3.0)
    except KeyboardInterrupt:
        print("\nInterrotto con Ctrl+C.")
    finally:
        det.close()
        cam.close()
        shutil.rmtree(out.parent, ignore_errors=True)
