"""CLI entry point for Qwen-ASR VR Chat.

Run ``python -m stt --help`` to see all options.
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

from stt.audio import SAMPLE_RATE, AudioProcessor, _Chunk
from stt.broadcast import BroadcastManager
from stt.config import load_config
from stt.obs import app as browser_app
from stt.obs import run_server as _run_browser_server
from stt.obs import server as browser_server
from stt.transcription import (
    TRANSCRIPTION_INTERVAL,
    TranscriptionBuffer,
    TranscriptionRunner,
)
from stt.twitch import TwitchChatClient

logger = logging.getLogger(__name__)
trans_logging.set_verbosity_error()
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

app = typer.Typer(help="VRChat speech-to-text (STT) via Qwen-ASR")


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
        self._audio_queue = queue.Queue()
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
        logger.info("Starting VRChat Speech-to-Text...")

        self.is_running = True

        # Shared stop event — controls whichever thread is running (Twitch or OSC)
        stop_event = threading.Event()

        # Start processing thread
        processing_thread = threading.Thread(target=self._process_loop)
        processing_thread.start()

        if self._twitch_client is not None:
            # Twitch mode: start chat thread + browser source
            twitch_thread = threading.Thread(
                target=self._twitch_client.run, args=(stop_event,)
            )
            twitch_thread.start()

            browser_app.state.broadcast_manager = self._broadcast_manager
            browser_source_thread = threading.Thread(target=_run_browser_server)
            browser_source_thread.start()
        else:
            # VRC/OSC mode: start OSC input thread (no browser source)
            osc_thread = threading.Thread(
                target=self._broadcast_manager.run, args=(stop_event,)
            )
            osc_thread.start()
            twitch_thread = None
            browser_source_thread = None

        # Start audio capture
        try:
            self._audio_processor.start()
            logger.info("Listening... (speak into your microphone)")

            while self.is_running:
                time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("Shutting down all threads...")
        finally:
            self._cleanup(
                processing_thread=processing_thread,
                twitch_thread=twitch_thread,
                browser_source_thread=browser_source_thread,
                stop_event=stop_event,
            )

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

            if not isinstance(buffer, list) or len(buffer) == 0:
                continue

            # Speech still ongoing — transcribe incrementally with interval
            self._process_ongoing_speech(buffer)

    def _process_ongoing_speech(self, buffer: _Chunk) -> None:
        """Handle speech that's still ongoing (accumulate and transcribe).

        Uses while/else pattern:
        - `break` from erase detected → skips else clause entirely
        - condition becomes False (None sentinel) → runs else clause with re-transcription
        """
        next_transcription = time.time() + TRANSCRIPTION_INTERVAL

        while buffer[-1] is not None:
            # At this point buffer contains no None values, so concat is safe.
            audio_arrays: list[np.ndarray] = [
                item for item in buffer if isinstance(item, np.ndarray)
            ]
            self._transcription_runner.process_chunk(np.concatenate(audio_arrays))

            # TranscriptionRunner's internal _is_erase_keyword() cleared buffer → stop.
            if not self._transcription_runner.transcription_buffer.intermediate:
                break  # Erase detected — skip else clause

            now = time.time()
            time_to_wait = max(0, next_transcription - now)
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            next_transcription = now + TRANSCRIPTION_INTERVAL
        else:
            # Natural exit via None sentinel (like old code's else clause).
            # Re-transcribe any items added between last iteration and None append.
            remaining: list[np.ndarray] = [
                item for item in buffer[:-1] if isinstance(item, np.ndarray)
            ]
            self._transcription_runner.process_chunk(np.concatenate(remaining))

        self._on_speech_end()

    def _on_speech_end(self) -> None:
        """Handle the end of a speech segment.

        TranscriptionRunner's internal _is_erase_keyword() handles erase detection
        mid-stream. This method delegates commit logic and notifies broadcast manager.
        """
        # If TranscriptionRunner already cleared buffer on erase detection,
        # intermediate will be empty → no commit needed.
        if not self._transcription_runner.transcription_buffer.intermediate:
            return

        # Commit to buffer, then notify broadcast manager (which selects text
        # based on mode: .last() for Twitch, .get() for OSC, .get_full() for clipboard)
        self._transcription_runner.commit_segment()

    def _cleanup(
        self,
        *,
        processing_thread: threading.Thread,
        twitch_thread: threading.Thread | None,
        browser_source_thread: threading.Thread | None,
        stop_event: threading.Event,
    ) -> None:
        """Gracefully shut down all components."""
        self.is_running = False
        self._audio_processor.stop()

        processing_thread.join()

        # Signal the shared stop event — stops whichever thread is running
        stop_event.set()

        if twitch_thread is not None:
            logger.debug("Stopping browser source thread")
            browser_server.should_exit = True
            if browser_source_thread is not None:
                browser_source_thread.join()

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
) -> None:
    """Start VRChat speech-to-text.

    In default mode, results are sent to VRC via OSC text boxes.
    With `--twitch`, results are sent to Twitch chat instead.
    Add `--clipboard` to also copy transcriptions to the system clipboard.
    """
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
