"""
Single-window MuJoCo + ImGui chat (bottom-right). Uses GLFW + OpenGL + mjr_render.

Run: python -m franka_emika_panda.integrated_viewer
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import traceback
from typing import Optional, Tuple

import glfw
import imgui
import mujoco
from imgui.integrations.glfw import GlfwRenderer
from OpenGL.GL import (
    GL_ALL_ATTRIB_BITS,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_FRAMEBUFFER,
    GL_LINK_STATUS,
    glBindFramebuffer,
    glClear,
    glGetProgramiv,
    glIsProgram,
    glPopAttrib,
    glPushAttrib,
    glViewport,
)

from franka_emika_panda.vla_controller import (
    capture_workspace_image,
    execute_plan,
    get_vla_plan,
    load_vla_scene,
    sanitize_execution_plan,
)


class IntegratedViewer:
    """GLFW + MuJoCo mjr + ImGui; `sync()` matches passive viewer refresh for motion code."""

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, renderer, script_dir: str) -> None:
        self.m = m
        self.d = d
        self.renderer = renderer
        self.script_dir = script_dir

        self._planning = False
        self._executing = False
        self._pending_cmd = ""
        self._cmd_queue: queue.Queue[str] = queue.Queue()
        self._result_queue: queue.Queue[Tuple[str, object]] = queue.Queue()

        self._cmd_text = ""
        self._status = "Enter a command and click Send (or Enter)."

        self._thread: Optional[threading.Thread] = None

        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        self._glfw_owned = True

        # MuJoCo MjrContext uses legacy GL (e.g. display lists); core profile omits those and
        # fails with "Could not allocate display lists". Use OpenGL 3.3 compatibility profile.
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)

        self.window = glfw.create_window(
            1280,
            720,
            "MuJoCo VLA — franka_vla_workspace",
            None,
            None,
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # Build MuJoCo GL resources before ImGui. If GlfwRenderer runs first, MjrContext can
        # invalidate ImGui's shader program (glUseProgram GL_INVALID_VALUE after mjr_render).
        self.ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.scn = mujoco.MjvScene(m, maxgeom=10000)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()

        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "main_cam")
        if cam_id < 0:
            raise RuntimeError("Camera 'main_cam' not found in model")
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = cam_id

        imgui.create_context()
        self.impl = GlfwRenderer(self.window)

    def _ensure_imgui_gl(self) -> None:
        """MuJoCo mjr_* can invalidate GL programs; also ensure link succeeded (else glUseProgram fails)."""
        h = getattr(self.impl, "_shader_handle", 0)
        if h and glIsProgram(h):
            try:
                st = glGetProgramiv(h, GL_LINK_STATUS)
                linked = int(st[0]) if hasattr(st, "__getitem__") else int(st)
            except Exception:
                linked = 0
            if linked:
                return
        try:
            self.impl._invalidate_device_objects()
        except Exception:
            pass
        self.impl._create_device_objects()
        self.impl.refresh_font_texture()

    def is_running(self) -> bool:
        return not glfw.window_should_close(self.window)

    def sync(self) -> None:
        if glfw.window_should_close(self.window):
            return

        glfw.poll_events()
        self.impl.process_inputs()
        imgui.new_frame()
        self._draw_chat_ui()

        w, h = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, max(w, 1), max(h, 1))
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if w >= 1 and h >= 1:
            # Isolate MuJoCo GL state so mjr_render does not delete or break ImGui's shader program.
            pushed = False
            try:
                glPushAttrib(GL_ALL_ATTRIB_BITS)
                pushed = True
            except Exception:
                pass
            try:
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)

                mujoco.mjv_updateScene(
                    self.m,
                    self.d,
                    self.opt,
                    self.pert,
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL,
                    self.scn,
                )
                viewport = mujoco.MjrRect(0, 0, w, h)
                mujoco.mjr_render(viewport, self.scn, self.ctx)
                mujoco.mjr_restoreBuffer(self.ctx)
            finally:
                if pushed:
                    try:
                        glPopAttrib()
                    except Exception:
                        pass
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

        imgui.render()
        self._ensure_imgui_gl()
        self.impl.render(imgui.get_draw_data())
        glfw.swap_buffers(self.window)

    def _draw_chat_ui(self) -> None:
        ws_w, ws_h = glfw.get_window_size(self.window)
        imgui.set_next_window_position(
            ws_w - 16,
            ws_h - 16,
            imgui.ALWAYS,
            pivot_x=1.0,
            pivot_y=1.0,
        )
        imgui.set_next_window_size(440, 168, imgui.ALWAYS)

        flags = imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE
        imgui.begin("VLA chat", False, flags)

        imgui.text_wrapped(self._status)
        imgui.separator()

        _, self._cmd_text = imgui.input_text("##cmd", self._cmd_text, 512)

        busy = self._planning or self._executing
        send = imgui.button("Send")

        if send and not busy:
            text = self._cmd_text.strip()
            if text:
                if text.lower() in ("quit", "exit"):
                    glfw.set_window_should_close(self.window, True)
                else:
                    self._cmd_queue.put(text)
                    self._cmd_text = ""
                    self._status = "Queued…"

        imgui.end()

    def _start_planning(self, cmd: str) -> None:
        self._pending_cmd = cmd
        self._planning = True
        self._status = "Capturing scene…"

        path = capture_workspace_image(self.m, self.d, self.renderer, self.script_dir)

        def worker() -> None:
            try:
                plan = get_vla_plan(path, cmd)
                self._result_queue.put(("ok", plan))
            except Exception as e:
                self._result_queue.put(("err", str(e)))
            finally:
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except OSError:
                    pass

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        self._status = "Planning (VLA API)…"

    def _drain_plan_result(self) -> None:
        if not self._planning:
            return
        try:
            kind, data = self._result_queue.get_nowait()
        except queue.Empty:
            return

        self._planning = False
        self._thread = None

        if kind == "err":
            self._status = f"Error: {data}"
            return

        plan = data
        if not plan:
            self._status = "No actionable plan generated."
            return

        plan = sanitize_execution_plan(plan, self._pending_cmd, self.d)
        print(json.dumps(plan, indent=2))
        self._executing = True
        self._status = "Executing plan…"
        try:
            execute_plan(plan, self.m, self.d, self, self._pending_cmd)
            self._status = "Done. Enter another command."
        finally:
            self._executing = False

    def _try_start_next_command(self) -> None:
        if self._planning or self._executing:
            return
        try:
            cmd = self._cmd_queue.get_nowait()
        except queue.Empty:
            return
        self._start_planning(cmd)

    def run(self) -> None:
        for _ in range(200):
            mujoco.mj_step(self.m, self.d)
            self.sync()

        while self.is_running():
            self._drain_plan_result()
            self._try_start_next_command()
            if not self._executing:
                mujoco.mj_step(self.m, self.d)
            self.sync()

        self.impl.shutdown()
        glfw.destroy_window(self.window)
        if self._glfw_owned:
            glfw.terminate()


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Loading scene and creating GLFW window (separate from this console)…", flush=True)
    m, d, renderer = load_vla_scene(script_dir)
    viewer = IntegratedViewer(m, d, renderer, script_dir)
    print("Window should appear — check the taskbar if it is behind other apps.", flush=True)
    viewer.run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
