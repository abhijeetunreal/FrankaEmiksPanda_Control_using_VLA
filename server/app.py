"""
FastAPI app: WebSocket for MuJoCo frames, chat commands, pointer drag, and log streaming.
"""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from franka_emika_panda.sim_service import SimService

STATIC_DIR = ROOT / "web"
outbound: queue.Queue[dict] = queue.Queue(maxsize=500)
clients: set[WebSocket] = set()
sim: SimService | None = None
pump_task: asyncio.Task | None = None


def _enqueue_out(msg: dict):
    try:
        outbound.put_nowait(msg)
    except queue.Full:
        try:
            outbound.get_nowait()
        except queue.Empty:
            pass
        try:
            outbound.put_nowait(msg)
        except queue.Full:
            pass


def _on_frame(rgb):
    rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8).copy())
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    _enqueue_out({"type": "frame", "data": b64})


def _on_log(line: str):
    _enqueue_out({"type": "log", "line": line})


def _on_plan(plan: list):
    _enqueue_out({"type": "plan", "json": plan})


def _on_error(message: str):
    _enqueue_out({"type": "error", "message": message})


def _on_busy(busy: bool):
    _enqueue_out({"type": "busy", "value": busy})


def get_sim() -> SimService:
    global sim
    if sim is None:
        sim = SimService(
            on_frame=_on_frame,
            on_log=_on_log,
            on_plan=_on_plan,
            on_error=_on_error,
            on_busy=_on_busy,
        )
        sim.start()
    return sim


app = FastAPI(title="VLA MuJoCo Workspace")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global pump_task
    get_sim()
    pump_task = asyncio.create_task(_pump_outbound_loop())


@app.on_event("shutdown")
async def shutdown():
    global sim, pump_task
    if pump_task:
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass
        pump_task = None
    if sim:
        sim.stop()
        sim = None


async def _pump_outbound_loop():
    while True:
        try:
            msg = outbound.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.015)
            continue
        dead = []
        for ws in list(clients):
            try:
                await ws.send_text(json.dumps(msg))
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    s = get_sim()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            t = data.get("type")
            if t == "command":
                text = (data.get("text") or "").strip()
                if text:
                    s.enqueue_command(text)
            elif t == "pointer":
                kind = data.get("kind") or "move"
                s.enqueue_pointer(kind, float(data.get("u", 0)), float(data.get("v", 0)))
            elif t == "camera":
                op = str(data.get("op") or "")
                if op:
                    s.enqueue_camera(
                        op,
                        dx=float(data.get("dx", 0.0)),
                        dy=float(data.get("dy", 0.0)),
                        delta=float(data.get("delta", 0.0)),
                    )
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)


@app.get("/api/health")
async def health():
    return {"ok": True}


if (STATIC_DIR / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:

    @app.get("/")
    async def root_hint():
        return {
            "message": "Missing web/index.html. Clone the repo with the web/ folder intact.",
            "health": "/api/health",
            "ws": "/ws",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
