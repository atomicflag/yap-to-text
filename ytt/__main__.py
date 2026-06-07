"""CLI entry point for Yap To Text.

Run ``uv run python -m ytt --help`` to see all options.
"""

import logging
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import typer
from transformers import logging as trans_logging

from ytt.audio import SAMPLE_RATE, AudioProcessor, _Chunk
from ytt.broadcast import BroadcastManager
from ytt.config import load_config
from ytt.obs import app as browser_app
from ytt.obs import run_server as _run_browser_server
from ytt.obs import server as browser_server
from ytt.transcription import (
    TRANSCRIPTION_INTERVAL,
    TranscriptionBuffer,
    TranscriptionRunner,
)
from ytt.twitch import TwitchChatClient

logger = logging.getLogger(__name__)
trans_logging.set_verbosity_error()
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

app = typer.Typer(help="Yap To Text — real-time captions for VRChat and Twitch")


class STTOrchestrator:
    """Orchestrates audio capture, transcription, and broadcasting."""

    def __init__(
        self,
        *,
        broadcast_manager: BroadcastManager,
        transcription_runner: TranscriptionRunner,
        twitch_client: TwitchChatClient | None,
    ) -> None:
        """Initialize the orchestrator with its components.

        Args:
            broadcast_manager: Manages broadcasting transcriptions.
            transcription_runner: Handles transcription logic and buffering.
            twitch_client: Optional Twitch chat client for Twitch mode.
        """
        self.is_running = False
        self._audio_queue: queue.Queue[_Chunk] = queue.Queue()
        self._broadcast_manager = broadcast_manager
        self._transcription_runner = transcription_runner
        self._twitch_client = twitch_client

        # Create and wire our own AudioProcessor so on_chunk is bound before start().
        self._audio_processor: AudioProcessor = AudioProcessor(
            sample_rate=SAMPLE_RATE,
            on_chunk=self._on_audio_chunk,
            on_speech_end=self._broadcast_manager.on_speech_end,
        )

    def run(self) -> None:
        """Start capturing audio and processing transcriptions."""
        logger.info("Starting Yap To Text...")

        self.is_running = True

        # Shared stop event — controls whichever thread is running (Twitch or OSC)
        stop_event = threading.Event()

        # Collect all threads before starting any of them
        threads: list[threading.Thread] = []

        # Processing thread (always present)
        processing_thread = threading.Thread(target=self._process_loop)
        threads.append(processing_thread)

        if self._twitch_client is not None:
            # Twitch mode: chat thread + browser source
            twitch_thread = threading.Thread(
                target=self._twitch_client.run, args=(stop_event,)
            )
            threads.append(twitch_thread)

            browser_app.state.broadcast_manager = self._broadcast_manager
            browser_source_thread = threading.Thread(target=_run_browser_server)
            threads.append(browser_source_thread)
        else:
            # VRC/OSC mode: OSC input thread only
            osc_thread = threading.Thread(
                target=self._broadcast_manager.run, args=(stop_event,)
            )
            threads.append(osc_thread)

        # Start all threads uniformly
        for t in threads:
            t.start()

        # Start audio capture
        try:
            self._audio_processor.start()
            logger.info("Listening... (speak into your microphone)")

            while self.is_running:
                time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("Shutting down all threads...")
        finally:
            self._cleanup(threads, stop_event)

    def _on_audio_chunk(self, chunk_list: _Chunk) -> None:
        """Receive a chunk of audio from the AudioProcessor.

        The list contains numpy arrays and optionally ends with ``None``
        to mark speech end.
        """
        self._audio_queue.put(chunk_list)
        # Notify broadcast manager — it reads text from its buffer dependency
        self._broadcast_manager.on_speech_start()

    def _process_loop(self) -> None:
        """Pull audio chunks from the queue and transcribe them."""
        logger.info("Starting audio processing thread")

        while self.is_running:
            try:
                buffer = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not buffer:
                continue

            # Speech still ongoing — transcribe incrementally with interval
            self._process_ongoing_speech(buffer)

    def _process_ongoing_speech(self, buffer: _Chunk) -> None:
        """Handle speech that's still ongoing (accumulate and transcribe)."""
        next_transcription = time.time() + TRANSCRIPTION_INTERVAL

        while buffer[-1] is not None:
            # At this point buffer contains no None values, so concat is safe.
            chunk = np.concatenate(buffer)  # type: ignore
            if self._transcription_runner.process_chunk(chunk):
                return
            now = time.time()
            time_to_wait = next_transcription - now
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            next_transcription = now + TRANSCRIPTION_INTERVAL
        # Re-transcribe any items added between last iteration and None append.
        chunk = np.concatenate(buffer[:-1])  # type: ignore
        if self._transcription_runner.process_chunk(chunk):
            return
        self._transcription_runner.commit_segment()

    def _cleanup(
        self,
        threads: list[threading.Thread],
        stop_event: threading.Event,
    ) -> None:
        """Gracefully shut down all components."""
        self.is_running = False
        self._audio_processor.stop()

        # Signal the shared stop event — stops whichever mode-thread is running
        stop_event.set()

        browser_server.should_exit = True

        for t in threads:
            t.join()

        logger.info("All threads stopped, exiting.")


@app.command()
def main(
    twitch: bool = typer.Option(False, "--twitch", help="Enable Twitch chat mode"),
    clipboard: bool = typer.Option(
        False, "--clipboard", help="Copy completed transcriptions to clipboard"
    ),
    config_path: str = typer.Option(
        "config.json",
        "--config",
        help="Path to config.json (default: current directory)",
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase log verbosity"
    ),
) -> None:
    """Yap To Text — real-time captions for VRChat and Twitch.

    In default mode, results are sent to VRC via OSC text boxes.
    With `--twitch`, results are sent to Twitch chat instead.
    Add `--clipboard` to also copy transcriptions to the system clipboard.
    """
    level = logging.INFO - (10 * min(verbose, 2))
    logging.getLogger().setLevel(level)
    config = load_config(Path(config_path))

    # Apply audio device settings from config (if configured)
    devices: tuple[str | None, str | None] = (
        config.audio_input_device,
        config.audio_output_device,
    )
    if all(devices):
        sd.default.device = devices

    # Wire components
    twitch_client: TwitchChatClient | None = None
    if twitch and config.twitch is not None:
        twitch_client = TwitchChatClient(
            app_id=config.twitch.app_id,
            app_secret=config.twitch.app_secret,
            refresh_token=config.twitch.refresh_token,
            target_channel=config.twitch.target_channel,
            config_path=Path(config_path),
        )

    # Shared buffer — passed to both BroadcastManager and TranscriptionRunner
    transcription_buffer = TranscriptionBuffer()

    broadcast_manager = BroadcastManager(
        transcription_buffer=transcription_buffer,
        twitch_mode=twitch,
        clipboard_mode=clipboard,
        twitch_chat=twitch_client,
    )

    transcription_runner: TranscriptionRunner = TranscriptionRunner(
        transcription_buffer=transcription_buffer,
        hallucinations=config.hallucinations,
        erase_keyword=config.erase_keyword,
        replacements=config.replacements,
        on_commit=broadcast_manager.on_commit,
        on_intermediate=broadcast_manager.on_intermediate,
        on_erase=broadcast_manager.on_erase,
    )

    # Streaming orchestrator (owns its own AudioProcessor)
    streamer = STTOrchestrator(
        broadcast_manager=broadcast_manager,
        transcription_runner=transcription_runner,
        twitch_client=twitch_client,
    )

    try:
        streamer.run()
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True, color=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    app()
