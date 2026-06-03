"""OBS Browser Source integration — serves live transcription data via FastAPI."""

import asyncio
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/transcript")
def transcript() -> dict:
    """Return the current transcription state for the OBS browser source.

    Returns a JSON object containing committed segments, typing status,
    and intermediate text.

    Returns:
        A dictionary with ``segments`` (list of [timestamp, text] pairs),
        ``is_typing`` (bool), and ``intermediate`` (current partial text).
    """
    broadcast_manager = app.state.broadcast_manager
    buf = broadcast_manager.transcription_buffer
    return {
        "segments": buf.get_segments(),
        "is_typing": broadcast_manager.speech_triggered,
        "intermediate": buf.intermediate,
    }


# Serve static files (HTML, JS, CSS) for the OBS browser source.
_static_dir = Path(__file__).parent.parent / "static"
app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")

config = uvicorn.Config(app=app, port=9098, log_level="info")
server = uvicorn.Server(config)


def run_server() -> None:
    """Run the browser source server in the calling thread (blocks)."""
    logger.info("Starting browser source thread")
    asyncio.run(server.serve())
