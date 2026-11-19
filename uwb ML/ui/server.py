"""Servidor de la interfaz (FastAPI). No contiene lógica de ML: solo reenvía eventos del contrato por WebSocket.

  python ui\\server.py --source sim
  python ui\\server.py --source csv --model results\\gb\\model.joblib --csv placa\\dataset_D07\\ml_dataset.csv
"""
import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

UI_DIR = Path(__file__).resolve().parent
CFG = json.loads((UI_DIR / "config.json").read_text(encoding="utf-8"))
ARGS = {"source": "sim", "model": None, "csv": None, "session": None}
clients = set()
last_event = None


async def pump():
    global last_event
    if ARGS["source"] == "csv":
        from sources.ml_adapter import events
        gen = events(CFG, ARGS["csv"], ARGS["model"], ARGS["session"])
    else:
        from sources.sim import events
        gen = events(CFG)
    async for ev in gen:
        last_event = ev
        text = json.dumps(ev, allow_nan=False)
        for ws in list(clients):
            try:
                await ws.send_text(text)
            except Exception:
                clients.discard(ws)


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(pump())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/config")
def get_config():
    return CFG


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


@app.get("/")
def index():
    return FileResponse(UI_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(UI_DIR))
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["sim", "csv"], default="sim")
    p.add_argument("--model", help=".joblib (solo con --source csv)")
    p.add_argument("--csv", help="ml_dataset.csv (solo con --source csv)")
    p.add_argument("--session")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    if a.source == "csv" and not (a.model and a.csv):
        p.error("--source csv necesita --model y --csv")
    ARGS.update(source=a.source, model=a.model, csv=a.csv, session=a.session)
    uvicorn.run(app, host=a.host, port=a.port)
