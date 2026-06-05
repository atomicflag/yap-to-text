"""Transcription processing: ASR inference, hallucination filtering, and buffering."""

import datetime
import logging
import string
import time
from collections.abc import Callable

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel

from .audio import SAMPLE_RATE

logger = logging.getLogger(__name__)

TRANSCRIPTION_INTERVAL: float = 1.0


class TranscriptionBuffer:
    """Accumulates and manages transcription segments with truncation support.

    Supports both committed (finalized) segments and intermediate (in-progress)
    text, with configurable character limits and inactivity timeouts.
    """

    def __init__(self, max_chars: int = 144, reset_timeout: float = 10.0) -> None:
        """Initialize the transcription buffer.

        Args:
            max_chars: Maximum character count for output. Longer text is truncated.
            reset_timeout: Seconds of inactivity before a new segment starts fresh.

        """
        self.max_chars: int = max_chars
        self.reset_timeout: float = reset_timeout
        self._text: str = ""
        self._last_update: float = 0.0
        self._segments: list[tuple[float, str]] = []
        self.intermediate: str = ""

    def push(self, new_text: str) -> None:
        """Push intermediate text into the buffer without committing."""
        if not new_text:
            return
        now: float = time.time()
        self.intermediate = new_text

        if (now - self._last_update) > self.reset_timeout:
            self._text = ""
            self._segments.clear()

        self._last_update = now

    def commit(self) -> None:
        """Commit intermediate text as a finalized segment."""
        if not self.intermediate:
            return

        new_text: str = self.intermediate.strip()
        if new_text[-1] not in string.punctuation:
            new_text += "."

        if self._text:
            self._text += " " + new_text
        else:
            self._text = new_text

        now: float = time.time()
        self._segments.append((now, new_text))
        self._last_update = now
        self.intermediate = ""

    def get(self, force_typing: bool = False) -> str:
        """Retrieve the current full transcription (possibly truncated).

        Args:
            force_typing: If True and no recent activity, shows ellipsis only.

        Returns:
            Truncated transcription text with an ellipsis if needed.

        """
        if (time.time() - self._last_update) > self.reset_timeout:
            self._text = ""
        text: str = (self._text + " " + self.intermediate).strip()

        if self.intermediate:
            text = text.rstrip(string.punctuation)
            text += "\u22ef"
        elif force_typing:
            text = (text + " \u22ef").strip()

        if len(text) <= self.max_chars:
            return text

        # Truncate to fit within max_chars, preserving word boundaries
        text = text[-self.max_chars + 2 :]
        first_space: int = text.index(" ")
        text = text[first_space + 1 :]
        return "\u2026 " + text

    def get_full(self) -> str:
        """Retrieve the full, untruncated transcription."""
        return (self._text + " " + self.intermediate).strip()

    def last(self) -> str | None:
        """Return the most recently committed segment text, or None."""
        if not self._segments:
            return None
        return self._segments[-1][1]

    def get_segments(self) -> list[tuple[float, str]]:
        """Return a snapshot of all committed segments with timestamps.

        Timestamps are UNIX epoch floats (seconds since epoch).
        """
        return list(self._segments)

    def clear(self) -> None:
        """Clear the buffer and all segments."""
        self._text = ""
        self._last_update = 0.0
        self._segments.clear()
        self.intermediate = ""


class TranscriptionRunner:
    """Processes audio chunks through ASR, filters hallucinations, and updates the buffer.

    Wraps Qwen3-ASR inference, applies hallucination and erase detection,
    and fires callbacks for commit, intermediate update, and erase events.
    """

    def __init__(
        self,
        *,
        transcription_buffer: TranscriptionBuffer,
        hallucinations: list[str],
        erase_keyword: str,
        on_commit: Callable[[], None],
        on_intermediate: Callable[[], None],
        on_erase: Callable[[], None],
    ) -> None:
        """Initialize the transcription runner.

        Args:
            transcription_buffer: Shared buffer for accumulating transcriptions.
            hallucinations: Strings to drop as ASR hallucinations.
            erase_keyword: Phrase that triggers a buffer clear when detected.
            on_commit: Callback fired when a segment is committed
            on_intermediate: Callback fired with intermediate updates
            on_erase: Callback fired when erase keyword is detected.

        """
        self.transcription_buffer = transcription_buffer

        # Load ASR model (heavy — done once at construction)
        self.transcriber = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B",
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
            max_inference_batch_size=32,
            max_new_tokens=256,
        )

        self.hallucinations: list[str] = hallucinations
        self.erase_keyword: str = erase_keyword
        self._on_commit: Callable[[], None] = on_commit
        self._on_intermediate: Callable[[], None] = on_intermediate
        self._on_erase: Callable[[], None] = on_erase

    def process_chunk(self, audio_array: np.ndarray) -> bool:
        """Process a single audio chunk through the ASR model.

        Args:
            audio_array: Audio data as a numpy array (mono, 16 kHz).

        """
        try:
            result = self.transcriber.transcribe(
                audio=(audio_array, SAMPLE_RATE),
                language="English",
            )

            new_text: str = result[0].text.strip()

            # Filter hallucinations
            if (new_text in self.hallucinations) or (
                len(new_text) == 2 and new_text[1] == "."
            ):
                logger.debug("Dropping hallucinated transcription: %r", new_text)
                return False

            if not new_text:
                return False

            # Check erase keyword before notifying — fires after push so buffer
            # reflects current state; on_erase clears intermediate when detected
            if self._is_erase_keyword(new_text):
                return True

            # Push to buffer
            self.transcription_buffer.push(new_text)
            self._on_intermediate()
        except Exception:
            logger.exception("Error transcribing")
        return False

    def commit_segment(self) -> None:
        """Commit the current intermediate text as a final segment."""
        if not self.transcription_buffer.intermediate:
            self._on_intermediate()
            return

        self.transcription_buffer.commit()

        timestamp: str = datetime.datetime.now().strftime("%H:%M:%S")
        displayed_text: str | None = self.transcription_buffer.last()
        if displayed_text:
            logger.info("[%s] %s", timestamp, displayed_text)

        # Fire commit callback — BroadcastManager reads from its buffer dependency
        # and selects .last() (Twitch), .get() (OSC), or .get_full() (clipboard)
        self._on_commit()

    def _is_erase_keyword(self, text: str) -> bool:
        """Check if the intermediate text matches the configured erase keyword.

        Clears the transcription buffer and fires the ``on_erase`` callback
        when a match is found.

        Returns:
            True if the erase keyword was detected, False otherwise.
        """
        cleaned: str = text.lower().strip(".")
        if cleaned == self.erase_keyword:
            logger.info("Erase keyword detected — clearing buffer.")
            self.transcription_buffer.clear()
            self._on_erase()
            return True
        return False
