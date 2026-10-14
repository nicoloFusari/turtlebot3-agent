"""Ponte tra l'agente e ROS 2: navigazione con Nav2, giro di ispezione con il detector, riconoscimento dei volti durante la navigazione, camera e cicalino del TurtleBot3."""
import contextlib
import math
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose, Spin
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import CompressedImage
from turtlebot3_msgs.srv import Sound

PLACES_FILE = Path(__file__).resolve().parent / "places.yaml"
IMAGE_TOPIC = "/camera/image_raw/compressed"
SHOTS_DIR = Path(__file__).resolve().parent / "inspect_shots"
NAV_RETRIES = 1
RETRY_PAUSE_S = 2.0
NEAR_GOAL_M = 0.6
MIN_SPIN = math.radians(5)
STEP_DEG = 20.0
LOOK_S = 1.0
CAMERA_HFOV_DEG = 54.0
BEEP_SOUND = 1

STATUS_NAMES = {
    GoalStatus.STATUS_SUCCEEDED: "succeeded",
    GoalStatus.STATUS_CANCELED: "canceled",
    GoalStatus.STATUS_ABORTED: "failed",
}


def _yaw_from_quat(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class RosBridge:
    def __init__(self, places_file=PLACES_FILE, image_topic=IMAGE_TOPIC,
                 detector=None, verifier=None, memory=None, max_checks=3):
        self.places_file = Path(places_file)
        self.detector = detector
        self.verifier = verifier
        self.memory = memory
        self.max_checks = max_checks

        self._pose = None
        self._goal_handle = None
        self._wake = None
        self._spun = 0.0
        self._stop_requested = threading.Event()

        self._img_cond = threading.Condition()
        self._img_msg = None
        self._img_seq = 0

        rclpy.init()
        self.node = Node("agent_bridge")

        amcl_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._on_pose, amcl_qos)

        self.node.create_subscription(
            CompressedImage, image_topic, self._on_image, qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup())

        self._nav = ActionClient(self.node, NavigateToPose, "navigate_to_pose")
        self._spin = ActionClient(self.node, Spin, "spin")
        self._sound = self.node.create_client(Sound, "/sound")

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def _on_pose(self, msg):
        self._pose = msg.pose.pose

    def _on_image(self, msg):
        with self._img_cond:
            self._img_msg = msg
            self._img_seq += 1
            self._img_cond.notify_all()

    def _load(self):
        return yaml.safe_load(self.places_file.read_text())

    def _lookup(self, place, viewpoint):
        places = self._load()["places"]
        if place not in places:
            raise KeyError(f"luogo '{place}' inesistente. Disponibili: {list(places)}")
        entry = places[place]
        if viewpoint is None:
            return entry["pose"]
        vps = entry.get("viewpoints", {})
        if viewpoint not in vps:
            raise KeyError(f"viewpoint '{viewpoint}' inesistente in '{place}'. "
                           f"Disponibili: {list(vps)}")
        return vps[viewpoint]

    def _make_goal_pose(self, pose):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position.x = float(pose["x"])
        msg.pose.position.y = float(pose["y"])
        msg.pose.orientation.z = math.sin(pose["yaw"] / 2.0)
        msg.pose.orientation.w = math.cos(pose["yaw"] / 2.0)
        return msg

    def _wait(self, future, timeout, cancel=None):
        done = threading.Event()
        self._wake = done
        future.add_done_callback(lambda _: done.set())
        if cancel is not None and cancel.is_set():
            done.set()
        done.wait(timeout)
        self._wake = None
        return future.done() and not self._stop_requested.is_set()

    def _run_action(self, client, goal, timeout, name, feedback=None, cancel=None):
        if not client.wait_for_server(timeout_sec=5.0):
            return "failed", f"{name} non risponde"

        send_future = client.send_goal_async(goal, feedback_callback=feedback)
        if not self._wait(send_future, 10.0):
            return "failed", "goal non accettato in tempo"
        handle = send_future.result()
        if not handle.accepted:
            return "failed", "goal rifiutato"
        self._goal_handle = handle

        result_future = handle.get_result_async()
        finished = self._wait(result_future, timeout, cancel)
        self._goal_handle = None

        if cancel is not None and cancel.is_set() and not result_future.done():
            handle.cancel_goal_async()
            return "canceled", "annullato"
        if not finished:
            handle.cancel_goal_async()
            return "canceled", ("stop richiesto" if self._stop_requested.is_set()
                                else "timeout")
        return STATUS_NAMES.get(result_future.result().status, "failed"), None

    def _rotate(self, angle_rad, timeout=120.0):
        goal = Spin.Goal()
        goal.target_yaw = float(angle_rad)
        goal.time_allowance = Duration(sec=int(timeout))
        self._spun = 0.0

        def on_feedback(msg):
            self._spun = abs(msg.feedback.angular_distance_traveled)

        return self._run_action(self._spin, goal, timeout + 5.0, "spin",
                                feedback=on_feedback)

    def _latest_frame(self, after=-1):
        with self._img_cond:
            if not self._img_cond.wait_for(
                    lambda: self._img_msg is not None and self._img_seq > after, 0.2):
                return None
            seq, msg = self._img_seq, self._img_msg
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        return None if frame is None else (seq, frame)

    def camera_jpeg(self):
        with self._img_cond:
            if self._img_msg is None:
                return None
            return self._img_seq, bytes(self._img_msg.data)

    def _wake_up(self):
        wake = self._wake
        if wake is not None:
            wake.set()

    def go_to(self, place, viewpoint=None, timeout=300.0):
        self._stop_requested.clear()
        try:
            pose = self._lookup(place, viewpoint)
        except KeyError as e:
            return {"status": "failed", "reason": e.args[0]}

        goal = NavigateToPose.Goal()
        goal.pose = self._make_goal_pose(pose)

        out = {"place": place, "viewpoint": viewpoint}
        for attempt in range(1 + NAV_RETRIES):
            if attempt:
                if self._stop_requested.wait(RETRY_PAUSE_S):
                    status, reason = "canceled", "stop richiesto"
                    break
                goal.pose = self._make_goal_pose(pose)
                out["retries"] = attempt
            status, reason, dist, near = self._navigate(goal, pose, timeout)
            if near or status != "failed":
                break
        out["status"] = status
        if near:
            out["status"] = "succeeded"
            out["note"] = (f"fermo a {dist:.2f} m dal punto: Nav2 non riusciva ad "
                           "arrivare più vicino, ma da qui si può osservare")
        elif reason:
            out["reason"] = reason
        return out

    def _navigate(self, goal, pose, timeout):
        near = threading.Event()
        state = {"dist": None, "recoveries": None}

        def on_feedback(msg):
            fb = msg.feedback
            pos = fb.current_pose.pose.position
            state["dist"] = math.hypot(pos.x - pose["x"], pos.y - pose["y"])
            if state["dist"] > NEAR_GOAL_M:
                state["recoveries"] = None
            elif state["recoveries"] is None:
                state["recoveries"] = fb.number_of_recoveries
            elif fb.number_of_recoveries > state["recoveries"] and not near.is_set():
                near.set()
                self._wake_up()

        status, reason = self._run_action(self._nav, goal, timeout, "Nav2",
                                          feedback=on_feedback, cancel=near)
        dist = state["dist"]
        stopped_near = near.is_set() and status == "canceled"
        failed_near = status == "failed" and dist is not None and dist <= NEAR_GOAL_M
        return status, reason, dist, stopped_near or failed_near

    def _shift_ignored(self, boxes, angle_rad):
        got = self._latest_frame() if boxes else None
        if got is None:
            return []
        width = got[1].shape[1]
        dx = width * math.degrees(angle_rad) / CAMERA_HFOV_DEG
        return [[b[0] + dx, b[1], b[2] + dx, b[3]] for b in boxes if b[0] + dx < width]

    def inspect(self, query, who=None, sweep=True, step_deg=STEP_DEG, look_s=LOOK_S):
        try:
            return self._inspect(query, who, sweep, step_deg, look_s)
        finally:
            if self.detector is not None:
                self.detector.clear_view()

    @contextlib.contextmanager
    def _watching(self, query, run_dir):
        self.detector.start(query, self._latest_frame, run_dir, watch_only=True)
        try:
            yield
        finally:
            self.detector.stop(keep_view=True)

    def _inspect(self, query, who, sweep, step_deg, look_s):
        self._stop_requested.clear()
        run_dir = SHOTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        total = 2 * math.pi if sweep else 0.0
        step = math.radians(step_deg)
        swept, frames = 0.0, 0

        def coverage():
            return {"rotation_deg": round(math.degrees(swept)),
                    "frames_analyzed": frames, "folder": str(run_dir)}

        if self.detector is None:
            return {"status": "failed", "reason": "nessun detector configurato",
                    "coverage": coverage()}
        known = self.memory.names(query) if self.memory else []
        if who is not None and who not in known:
            return {"status": "failed", "coverage": coverage(),
                    "reason": f"in memoria non c'è '{query}' di '{who}'. "
                              f"Nomi disponibili per '{query}': {known}"}

        self.detector.prepare(query)
        found = threading.Event()
        ignore, rejected, others = [], [], []
        hit, best, check, error = None, None, None, None
        while True:
            found.clear()
            self.detector.start(query, self._latest_frame, run_dir, found.set, ignore)
            try:
                found.wait(look_s)
            finally:
                result = self.detector.stop(keep_view=True)
            frames += result["frames"]
            ignore = result["ignore"]
            b = result["best"]
            if b and (best is None or b["confidence"] > best["confidence"]):
                best = b

            hit = result["hit"]
            if hit is not None:
                verify = self.verifier and query not in self.detector.overrides
                if verify:
                    with self._watching(query, run_dir):
                        check = self.verifier(query, hit)
                else:
                    check = None
                if check is not None and check.get("confirmed") is False:
                    rejected.append({**hit, "reasoning": check.get("reasoning")})
                elif known:
                    if "identity" not in hit:
                        hit["identity"] = self.memory.identify(query, hit)
                    if who is None or hit["identity"]["name"] == who:
                        break
                    others.append(hit)
                else:
                    break
                ignore.append(hit["box"])
                hit = None
                if len(rejected) + len(others) >= self.max_checks:
                    break
                continue
            if self._stop_requested.is_set():
                error = {"status": "canceled", "reason": "stop richiesto"}
                break
            if frames == 0:
                error = {"status": "failed",
                         "reason": "nessuna immagine ricevuta dalla camera"}
                break
            if total - swept <= MIN_SPIN:
                break

            with self._watching(query, run_dir):
                status, reason = self._rotate(min(step, total - swept))
            swept += self._spun
            if self._stop_requested.is_set():
                error = {"status": "canceled", "reason": "stop richiesto"}
                break
            if status != "succeeded":
                error = {"status": "failed",
                         "reason": f"rotazione non riuscita: {reason or status}"}
                break
            ignore = self._shift_ignored(ignore, self._spun)

        if hit:
            self.beep()
            out = {"query": query, "status": "found", "detections": [hit],
                   "image": hit["image"], "coverage": coverage(),
                   "where": self.where_am_i()}
            if check is not None:
                out["vlm_check"] = check
            if rejected:
                out["rejected"] = rejected
            if others:
                out["others"] = others
            return out
        if error is not None:
            return {**error, "coverage": coverage()}
        out = {"query": query,
               "status": "uncertain" if best or rejected else "not_found",
               "detections": [best] if best else [],
               "coverage": coverage(), "where": self.where_am_i()}
        if rejected:
            out["rejected"] = rejected
        if others:
            out["others"] = others
        return out

    def beep(self, sound=BEEP_SOUND):
        def run():
            if self._sound.wait_for_service(timeout_sec=1.0):
                self._sound.call_async(Sound.Request(value=sound))
        threading.Thread(target=run, daemon=True).start()

    def where_am_i(self):
        if self._pose is None:
            return {"error": "nessuna posa su /amcl_pose: AMCL è inizializzato?"}
        x, y = self._pose.position.x, self._pose.position.y
        nearest, best = None, float("inf")
        for name, entry in self._load()["places"].items():
            p = entry["pose"]
            d = math.hypot(p["x"] - x, p["y"] - y)
            if d < best:
                nearest, best = name, d
        return {"x": round(x, 2), "y": round(y, 2),
                "yaw": round(_yaw_from_quat(self._pose.orientation), 2),
                "nearest_place": nearest, "distance_m": round(best, 2)}

    def get_map(self):
        data = self._load()
        rooms = {name: {"viewpoints": list(entry.get("viewpoints", {}))}
                 for name, entry in data["places"].items()}
        connections = [[c["a"], c["b"]] for c in data.get("connections", [])]
        return {"rooms": rooms, "connections": connections}

    def stop(self):
        self._stop_requested.set()
        self._wake_up()
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()

    def shutdown(self):
        self._executor.shutdown()
        self._spin_thread.join(2.0)
        self.node.destroy_node()
        rclpy.shutdown()
