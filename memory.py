"""Memoria locale di volti e oggetti noti: dice di chi è ciò che il robot trova (volti con YuNet e SFace, oggetti con CLIP). Uso: python3 memory.py add, list o check"""
import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
KNOWN_DIR = HERE / "known"
MODELS_DIR = HERE / "models"

FACE_LABEL = "person"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
YUNET = "face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface/face_recognition_sface_2021dec.onnx"

FACE_THRESHOLD = 0.363
OBJECT_THRESHOLD = 0.90
FACE_MAX_W = 960


def _zoo_model(path):
    file = MODELS_DIR / Path(path).name
    if not file.exists():
        MODELS_DIR.mkdir(exist_ok=True)
        print(f"Scarico {file.name}...")
        urllib.request.urlretrieve(ZOO + path, file)
    return str(file)


class Memory:
    def __init__(self, known_dir=KNOWN_DIR):
        self.dir = Path(known_dir)
        self._face_models = None
        self._clip = None
        self._vectors = {}

    def labels(self):
        if not self.dir.is_dir():
            return []
        return sorted(d.name for d in self.dir.iterdir() if self.names(d.name))

    def names(self, label):
        d = self.dir / label
        if not d.is_dir():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir() and any(p.glob("*.png")))

    def threshold(self, label):
        return FACE_THRESHOLD if label == FACE_LABEL else OBJECT_THRESHOLD

    def add(self, label, name, crop, frame=None, box=None):
        folder = self.dir / label / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{datetime.now():%Y%m%d_%H%M%S_%f}.png"
        cv2.imwrite(str(path), crop)
        if frame is not None:
            cv2.imwrite(str(path.with_suffix(".ref.jpg")), frame)
            path.with_suffix(".json").write_text(json.dumps({"box": box}))
        self._vectors.pop(label, None)
        return path

    def _faces(self):
        if self._face_models is None:
            self._face_models = (
                cv2.FaceDetectorYN.create(_zoo_model(YUNET), "", (320, 320)),
                cv2.FaceRecognizerSF.create(_zoo_model(SFACE), ""))
        return self._face_models

    def faces_in(self, frame):
        detector, recognizer = self._faces()
        h, w = frame.shape[:2]
        scale = min(1.0, FACE_MAX_W / w)
        small = cv2.resize(frame, (round(w * scale), round(h * scale))) if scale < 1 else frame
        detector.setInputSize((small.shape[1], small.shape[0]))
        faces = detector.detect(small)[1]
        if faces is None:
            return []
        faces = sorted(faces, key=lambda f: -f[2] * f[3])
        return [{"box": [round(float(v) / scale) for v in (f[0], f[1], f[0] + f[2], f[1] + f[3])],
                 "score": float(f[14]), "crop": recognizer.alignCrop(small, f)}
                for f in faces]

    def face_crops(self, frame):
        return [f["crop"] for f in self.faces_in(frame)]

    def detect_faces(self, frame):
        known = bool(self.names(FACE_LABEL))
        detections, annotated = [], frame.copy()
        for face in self.faces_in(frame):
            det = {"label": FACE_LABEL, "confidence": round(face["score"], 2),
                   "box": face["box"]}
            text = f"volto {det['confidence']:.2f}"
            if known:
                name, score = self.nearest(FACE_LABEL, self.embed(FACE_LABEL, face["crop"]))
                det["identity"] = {"name": name if score >= FACE_THRESHOLD else None,
                                   "score": round(score, 2)}
                text = f"{det['identity']['name'] or '?'} {score:.2f}"
            detections.append(det)
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 3)
            cv2.putText(annotated, text, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 200, 0), 2)
        detections.sort(key=lambda d: -d["confidence"])
        return detections, annotated

    def embed(self, label, crop):
        if label == FACE_LABEL:
            vec = self._faces()[1].feature(crop)[0]
        else:
            import clip
            import torch
            from PIL import Image
            if self._clip is None:
                self._clip = clip.load("ViT-B/32", device="cpu",
                                       download_root=str(MODELS_DIR))
            model, preprocess = self._clip
            image = preprocess(Image.fromarray(crop[:, :, ::-1])).unsqueeze(0)
            with torch.no_grad():
                vec = model.encode_image(image)[0].numpy()
        return vec / np.linalg.norm(vec)

    def vectors(self, label):
        if label not in self._vectors:
            names, vecs = [], []
            for name in self.names(label):
                for path in sorted((self.dir / label / name).glob("*.png")):
                    names.append(name)
                    vecs.append(self.embed(label, cv2.imread(str(path))))
            self._vectors[label] = (names, np.array(vecs))
        return self._vectors[label]

    def nearest(self, label, vec):
        names, vecs = self.vectors(label)
        if not names:
            return None, 0.0
        scores = vecs @ vec
        i = int(scores.argmax())
        return names[i], float(scores[i])

    def identify(self, label, hit):
        if not self.names(label):
            return None
        crop = cv2.imread(hit["crop_image"])
        name, score = self.nearest(label, self.embed(label, crop))
        return {"name": name if score >= OBJECT_THRESHOLD else None,
                "score": round(score, 2)}


def _add(memory, args):
    detector = None
    if args.label != FACE_LABEL:
        from detector import LiveDetector, masked_crop
        detector = LiveDetector(show=False)

    def select(frame):
        x, y, w, h = cv2.selectROI(f"{args.label} di {args.name}: trascina il "
                                   "riquadro, Invio conferma, c salta", frame,
                                   showCrosshair=False)
        cv2.destroyAllWindows()
        return [x, y, x + w, y + h] if w and h else None

    def crop_of(frame):
        if detector is None:
            crops = memory.face_crops(frame)
            return (crops[0], None) if crops else None
        if args.select:
            box = select(frame)
            if box is None:
                return None
            det = {"box": box, "mask": detector.segment(frame, box)}
            if det["mask"] is None:
                print("  (contorno non trovato: ritaglio senza maschera)")
            return masked_crop(frame, det), box
        detections = detector.detect(args.label, frame)[0]
        return (masked_crop(frame, detections[0]), None) if detections else None

    def save(got, frame, source):
        crop, box = got
        path = memory.add(args.label, args.name, crop,
                          frame if box else None, box)
        print(f"  {source} -> {path.relative_to(HERE)}")

    if args.files:
        for file in args.files:
            frame = cv2.imread(file)
            got = None if frame is None else crop_of(frame)
            if got is None:
                print(f"  {file}: niente da salvare ('{args.label}' non trovato)")
            else:
                save(got, frame, file)
        return

    if args.robot:
        from ros_bridge import RosBridge
        source = RosBridge()
        get_frame, close = source._latest_frame, source.shutdown
    else:
        from detector import WebcamSource
        source = WebcamSource(args.camera)
        get_frame, close = source.get_frame, source.close
    print(f"Cerco '{args.label}' per {args.shots} foto di {args.name}: tra una foto "
          "e l'altra cambia un po' angolo e distanza.")
    try:
        seq, saved = -1, 0
        deadline = time.monotonic() + (float("inf") if args.select else 60.0)
        while saved < args.shots and time.monotonic() < deadline:
            got = get_frame(seq)
            if got is None:
                continue
            seq, frame = got
            result = crop_of(frame)
            if result is not None:
                saved += 1
                save(result, frame, f"foto {saved}/{args.shots}")
                if not args.select:
                    time.sleep(1.5)
        if saved < args.shots:
            print(f"Tempo scaduto: salvate {saved} foto su {args.shots}.")
    finally:
        close()


def _list(memory):
    for label in memory.labels():
        for name in memory.names(label):
            count = len(list((memory.dir / label / name).glob("*.png")))
            print(f"{label}/{name}: {count} foto")
    if not memory.labels():
        print("Memoria vuota.")


def _check(memory):
    for label in memory.labels():
        names, vecs = memory.vectors(label)
        names = np.array(names)
        scores = vecs @ vecs.T
        np.fill_diagonal(scores, -1.0)
        same = [scores[i][names == n].max() for i, n in enumerate(names)
                if (names == n).sum() > 1]
        other = [scores[i][names != n].max() for i, n in enumerate(names)
                 if (names != n).any()]
        print(f"{label} (soglia attuale {memory.threshold(label)}):")
        if same:
            print(f"  stesso nome:  da {min(same):.2f} a {max(same):.2f}")
        if other:
            print(f"  altro nome:   da {min(other):.2f} a {max(other):.2f}")
        if same and other:
            gap = min(same) - max(other)
            print(f"  scarto: {gap:+.2f}" + ("" if gap > 0 else
                  "  -> i nomi si sovrappongono: nessuna soglia li separa"))
        else:
            print("  servono almeno due nomi con due foto ciascuno")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Memoria di volti e oggetti noti")
    sub = ap.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="aggiunge foto di un volto o di un oggetto")
    add.add_argument("label", help="'person' per un volto, altrimenti l'etichetta "
                                   "del detector in inglese, es. 'keys'")
    add.add_argument("name", help="di chi è, es. vitto")
    add.add_argument("files", nargs="*", help="immagini; senza, usa una camera")
    add.add_argument("--camera", type=int, default=0, help="webcam del PC")
    add.add_argument("--robot", action="store_true", help="camera del robot")
    add.add_argument("--shots", type=int, default=5)
    add.add_argument("--select", action="store_true",
                     help="disegni tu il riquadro, per quando il detector non trova "
                          "l'oggetto da solo (es. le chiavi)")
    sub.add_parser("list", help="elenca cosa c'è in memoria")
    sub.add_parser("check", help="controlla se le soglie separano i nomi")
    args = ap.parse_args()

    memory = Memory()
    if args.command == "add":
        _add(memory, args)
    elif args.command == "list":
        _list(memory)
    else:
        _check(memory)
