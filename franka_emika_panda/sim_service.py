"""
Headless MuJoCo sim loop for the web workspace: frames, VLA commands, object drag.
Runs on a dedicated thread; WebSocket layer enqueues messages and consumes outbound queues.
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from io import StringIO
from typing import Callable, Optional

import mujoco
import numpy as np

from . import vla_controller as vc

CAMERA_NAME = "main_cam"
RENDER_H = 480
RENDER_W = 640
DRAG_PLANE_Z = 0.42
DRAGGABLE_BODIES = frozenset({"red_box", "blue_box", "green_box"})



def _camera_basis_from_orbit(
    lookat: np.ndarray, azimuth_deg: float, elevation_deg: float, distance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (position, forward, right, up) for an orbit camera."""
    az = np.radians(float(azimuth_deg))
    el = np.radians(float(elevation_deg))
    orbit = np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=np.float64,
    )
    pos = np.asarray(lookat, dtype=np.float64) + float(distance) * orbit
    forward = np.asarray(lookat, dtype=np.float64) - pos
    forward /= np.linalg.norm(forward) + 1e-12

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right /= np.linalg.norm(right) + 1e-12

    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-12
    return pos, forward, right, up



def _camera_ray_from_uv(
    lookat: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    fovy_deg: float,
    u: float,
    v: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized image coords (u,v) in [0,1], origin top-left; returns (pnt, vec)."""
    pos, forward, right, up = _camera_basis_from_orbit(lookat, azimuth_deg, elevation_deg, distance)
    fovy_rad = np.radians(float(fovy_deg))
    aspect = width / float(height)
    tan_half = np.tan(fovy_rad / 2.0)
    ndc_x = (2.0 * u - 1.0) * aspect * tan_half
    ndc_y = (1.0 - 2.0 * v) * tan_half
    dir_world = forward + ndc_x * right + ndc_y * up
    dir_world /= np.linalg.norm(dir_world)
    return pos, dir_world



def _mj_ray_first_geom(m: mujoco.MjModel, d: mujoco.MjData, pnt: np.ndarray, vec: np.ndarray) -> tuple[int, float]:
    geomid = np.zeros(1, dtype=np.int32)
    normal = np.zeros(3, dtype=np.float64)
    dist = float(mujoco.mj_ray(m, d, pnt, vec, None, 1, -1, geomid, normal))
    gid = int(geomid[0])
    return gid, dist



def _geom_to_body_name(m: mujoco.MjModel, geomid: int) -> Optional[str]:
    if geomid < 0:
        return None
    bid = int(m.geom_bodyid[geomid])
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)



def _ray_plane_intersection(pnt: np.ndarray, vec: np.ndarray, z_plane: float) -> Optional[np.ndarray]:
    if abs(vec[2]) < 1e-9:
        return None
    t = (z_plane - float(pnt[2])) / float(vec[2])
    if t <= 0:
        return None
    return pnt + t * vec



def _free_joint_qpos_slice(m: mujoco.MjModel, body_name: str) -> Optional[tuple[int, int]]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    jid = int(m.body_jntadr[bid])
    if jid < 0:
        return None
    if int(m.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
        return None
    qadr = int(m.jnt_qposadr[jid])
    return qadr, 7


class SimService:
    def __init__(
        self,
        xml_path: Optional[str] = None,
        on_frame: Optional[Callable[[np.ndarray], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_plan: Optional[Callable[[list], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_busy: Optional[Callable[[bool], None]] = None,
    ):
        self.xml_path = xml_path or vc.default_xml_path()
        self.on_frame = on_frame
        self.on_log = on_log
        self.on_plan = on_plan
        self.on_error = on_error
        self.on_busy = on_busy

        self._m, self._d = vc.init_mujoco_session(self.xml_path)
        self._renderer: Optional[mujoco.Renderer] = None
        self._free_camera: Optional[mujoco.MjvCamera] = None
        self._queue: queue.Queue[dict] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vla")
        self._pending_future: Optional[Future] = None
        self._pending_cmd: Optional[str] = None
        self._cmd_backlog: deque[str] = deque()
        self._drag_body: Optional[str] = None
        self._last_frame_t = 0.0
        self._frame_interval = 1.0 / 30.0
        self._busy = False

        self._camera_lookat = np.array([0.62, 0.0, 0.52], dtype=np.float64)
        self._camera_azimuth = 180.0
        self._camera_elevation = -22.0
        self._camera_distance = 1.35
        self._camera_fovy = 45.0

    def log(self, line: str):
        if self.on_log:
            self.on_log(line)

    def _set_busy(self, v: bool):
        self._busy = v
        if self.on_busy:
            self.on_busy(v)

    def _emit_error(self, message: str):
        self.log(message)
        if self.on_error:
            self.on_error(message)

    def _sync_camera(self):
        if self._free_camera is None:
            return
        self._free_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._free_camera.fixedcamid = -1
        self._free_camera.trackbodyid = -1
        self._free_camera.lookat[:] = self._camera_lookat
        self._free_camera.azimuth = self._camera_azimuth
        self._free_camera.elevation = self._camera_elevation
        self._free_camera.distance = self._camera_distance

    def _grab_rgb(self) -> np.ndarray:
        if self._renderer is None:
            raise RuntimeError("Renderer not initialized")
        if self._free_camera is None:
            raise RuntimeError("Free camera not initialized")
        self._sync_camera()
        self._renderer.update_scene(self._d, camera=self._free_camera)
        raw = self._renderer.render()
        return np.ascontiguousarray(np.asarray(raw, dtype=np.uint8).copy())

    def _emit_frame(self):
        now = time.monotonic()
        if now - self._last_frame_t < self._frame_interval:
            return
        self._last_frame_t = now
        rgb = self._grab_rgb()
        if self.on_frame:
            self.on_frame(rgb)

    def _frame_hook(self):
        self._emit_frame()

    def _capture_rgb(self) -> np.ndarray:
        return self._grab_rgb()

    def enqueue_command(self, text: str):
        self._queue.put({"type": "command", "text": text})

    def enqueue_pointer(self, kind: str, u: float = 0.0, v: float = 0.0):
        self._queue.put({"type": "pointer", "kind": kind, "u": float(u), "v": float(v)})

    def enqueue_camera(self, op: str, **kwargs):
        msg = {"type": "camera", "op": op}
        msg.update(kwargs)
        self._queue.put(msg)

    def start(self):
        if self._running:
            return
        self._running = True
        vc.set_frame_hook(self._frame_hook, stride=5)
        self._thread = threading.Thread(target=self._run_loop, name="mujoco-sim", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._queue.put({"type": "shutdown"})
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        vc.set_frame_hook(None)
        self._executor.shutdown(wait=False)
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    def _can_start_command(self) -> bool:
        return self._pending_future is None and not self._busy

    def _start_vla(self, text: str):
        self.log(f"\n[VLA] Command: {text}")
        self._set_busy(True)

        try:
            rgb = self._capture_rgb()
        except Exception as e:
            self._set_busy(False)
            self._emit_error(f"Render failed: {e}")
            self._pump_backlog()
            return

        def _call():
            buf = StringIO()
            with contextlib.redirect_stdout(buf):
                plan = vc.get_vla_plan(text, rgb=rgb)
            for line in buf.getvalue().splitlines():
                if line.strip():
                    self.log(line)
            return plan

        self._pending_future = self._executor.submit(_call)
        self._pending_cmd = text

    def _finish_vla(self):
        fut = self._pending_future
        self._pending_future = None
        cmd = self._pending_cmd or ""
        self._pending_cmd = None

        try:
            plan = fut.result()
        except Exception as e:
            self._emit_error(f"VLA request failed: {e}")
            self._set_busy(False)
            self._pump_backlog()
            return

        if plan:
            plan = vc.sanitize_execution_plan(plan, cmd, self._d)
            self.log(json.dumps(plan, indent=2))
            if self.on_plan:
                self.on_plan(plan)
            buf = StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    vc.execute_plan(plan, self._m, self._d, None, cmd)
            except Exception as e:
                self._emit_error(f"Execution failed: {e}")
            else:
                for line in buf.getvalue().splitlines():
                    if line.strip():
                        self.log(line)
        else:
            self.log("[!] No actionable plan generated.")

        self._set_busy(False)
        self._pump_backlog()

    def _pump_backlog(self):
        if self._cmd_backlog and self._can_start_command():
            t = self._cmd_backlog.popleft()
            self._start_vla(t)

    def _handle_pointer(self, msg: dict):
        kind = msg.get("kind")
        u = float(msg.get("u", 0.0))
        v = float(msg.get("v", 0.0))

        mujoco.mj_forward(self._m, self._d)
        try:
            pnt, vec = _camera_ray_from_uv(
                self._camera_lookat,
                self._camera_azimuth,
                self._camera_elevation,
                self._camera_distance,
                self._camera_fovy,
                u,
                v,
                RENDER_W,
                RENDER_H,
            )
        except Exception:
            return

        if kind == "down":
            gid, dist = _mj_ray_first_geom(self._m, self._d, pnt, vec)
            if dist < 0:
                self._drag_body = None
                return
            body = _geom_to_body_name(self._m, gid)
            if body and body in DRAGGABLE_BODIES:
                self._drag_body = body
                self.log(f"[Scene] Picked {body}")
            else:
                self._drag_body = None

        elif kind == "move" and self._drag_body:
            hit = _ray_plane_intersection(pnt, vec, DRAG_PLANE_Z)
            if hit is not None:
                sl = _free_joint_qpos_slice(self._m, self._drag_body)
                if sl:
                    qadr, _ = sl
                    self._d.qpos[qadr : qadr + 3] = hit
                    mujoco.mj_forward(self._m, self._d)

        elif kind == "up":
            self._drag_body = None

    def _handle_camera(self, msg: dict):
        op = str(msg.get("op") or "").strip()
        if op == "orbit":
            dx = float(msg.get("dx", 0.0))
            dy = float(msg.get("dy", 0.0))
            self._camera_azimuth += dx * 0.25
            self._camera_elevation = float(np.clip(self._camera_elevation + dy * 0.25, -89.0, 89.0))
        elif op == "zoom":
            delta = float(msg.get("delta", 0.0))
            factor = 1.0 + 0.1 * delta
            factor = float(np.clip(factor, 0.7, 1.3))
            self._camera_distance = float(np.clip(self._camera_distance * factor, 0.25, 5.0))
        elif op == "pan":
            dx = float(msg.get("dx", 0.0))
            dy = float(msg.get("dy", 0.0))
            _, _, right, up = _camera_basis_from_orbit(
                self._camera_lookat,
                self._camera_azimuth,
                self._camera_elevation,
                self._camera_distance,
            )
            scale = 0.0025 * self._camera_distance
            self._camera_lookat = self._camera_lookat - right * dx * scale + up * dy * scale
        elif op == "reset":
            self._camera_lookat = np.array([0.62, 0.0, 0.52], dtype=np.float64)
            self._camera_azimuth = 180.0
            self._camera_elevation = -22.0
            self._camera_distance = 1.35

    def _run_loop(self):
        try:
            self._renderer = mujoco.Renderer(self._m, height=RENDER_H, width=RENDER_W)
            self._camera_fovy = float(self._m.vis.global_.fovy)
            self._free_camera = mujoco.MjvCamera()
            self._sync_camera()
            for _ in range(12):
                self._renderer.update_scene(self._d, camera=self._free_camera)
                _ = np.asarray(self._renderer.render(), dtype=np.uint8).copy()
        except Exception as e:
            self._emit_error(f"Renderer initialization failed: {e}")
            self._running = False
            return

        while self._running:
            try:
                msg = self._queue.get(timeout=0.05)
            except queue.Empty:
                msg = None

            if msg and msg.get("type") == "shutdown":
                break

            if msg and msg.get("type") == "pointer":
                self._handle_pointer(msg)
                continue

            if msg and msg.get("type") == "camera":
                self._handle_camera(msg)
                continue

            if msg and msg.get("type") == "command":
                text = (msg.get("text") or "").strip()
                if text:
                    if self._can_start_command():
                        self._start_vla(text)
                    else:
                        self._cmd_backlog.append(text)
                continue

            if self._pending_future is not None:
                if self._pending_future.done():
                    self._finish_vla()
                else:
                    mujoco.mj_step(self._m, self._d)
                    self._emit_frame()
                continue

            mujoco.mj_step(self._m, self._d)
            self._emit_frame()
