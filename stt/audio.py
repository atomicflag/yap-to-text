"""Audio capture, speech detection, and VAD processing."""

import collections
import logging
from collections.abc import Callable

import numpy as np
import sounddevice as sd
import torch

logger = logging.getLogger(__name__)

# Constants

SAMPLE_RATE: int = 16000
FRAME_SIZE: int = 512
PRE_SPEECH_MS: int = 1000
SILENCE_DURATION_MS: int = 1000
MIN_SPEECH_DURATION_MS: int = 500
SENSITIVITY: float = 0.2
SILENCE_SENSITIVITY: float = 0.3

# Derived values (computed once at module load)
FRAME_DURATION_MS: int = int(SAMPLE_RATE / FRAME_SIZE)
MAX_SILENCE_FRAMES: int = int(SILENCE_DURATION_MS / FRAME_DURATION_MS)
MIN_SPEECH_FRAMES: int = int(MIN_SPEECH_DURATION_MS / FRAME_DURATION_MS)

# Type alias for chunk lists that may end with None sentinel
type _Chunk = list[np.ndarray | None]


class AudioProcessor:
    """Captures audio from the microphone and detects speech boundaries.

    Manages a sounddevice InputStream, runs Silero VAD on incoming frames,
    and produces chunk lists when speech starts or ends. Emits chunks via
    the ``on_chunk`` callback.
    """

    def __init__(
        self,
        *,
        on_chunk: Callable[[_Chunk], None],
        on_speech_end: Callable[[], None],
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        """Initialize the audio processor.

        Args:
            on_chunk: Callback invoked with a list of numpy arrays when speech
                      starts (includes pre-speech buffer) and each array at the
                      end of speech as a sentinel ``None`` marker.
            on_speech_end: Callback invoked when speech ends (silence threshold
                           exceeded).
            sample_rate: Audio sample rate in Hz. Defaults to 16000.

        """
        self.sample_rate: int = sample_rate
        self.on_chunk: Callable[[_Chunk], None] = on_chunk
        self._on_speech_end: Callable[[], None] = on_speech_end

        # Speech detection state
        self.pre_speech_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=PRE_SPEECH_MS // FRAME_DURATION_MS
        )
        # Mutable buffer — audio_callback appends np.ndarray during speech,
        # then appends None in-place on silence to signal end of segment.
        self.speech_buffer: _Chunk = []
        self.speech_triggered: bool = False
        self.silence_counter: int = 0
        self.speech_probability: collections.deque[float] = collections.deque(
            maxlen=MIN_SPEECH_FRAMES
        )

        # Thread management
        self._input_stream: sd.InputStream | None = None

    def start(self) -> None:
        """Start the microphone input stream."""
        self._load_vad_model()

        self._input_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            blocksize=FRAME_SIZE,
        )
        self._input_stream.start()

    def stop(self) -> None:
        """Stop the microphone input stream."""
        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None

    # Private helpers

    def _load_vad_model(self) -> None:
        """Load the Silero VAD model via torch.hub."""
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.vad: torch.nn.Module = model

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,  # type: ignore[type-arg]
        status: sd.CallbackFlags,
    ) -> None:
        """Sounddevice callback invoked for each audio block."""
        if status:
            logger.warning("Audio callback status: %s", status)

        speech_conf = self.vad(torch.from_numpy(indata[:, 0]), self.sample_rate).item()
        self.speech_probability.append(speech_conf)
        avg_prob = sum(self.speech_probability) / len(self.speech_probability)
        is_speech: bool = avg_prob > SENSITIVITY

        indata_copy: np.ndarray = indata.copy().flatten()

        if not self.speech_triggered:
            # Waiting for speech to start
            self.pre_speech_buffer.append(indata_copy)

            if is_speech:
                self._on_speech_started()
        else:
            # Already inside a speech segment
            is_speech = avg_prob > SILENCE_SENSITIVITY

            if not is_speech:
                self.silence_counter += 1
                self.pre_speech_buffer.append(indata_copy)

                if self.silence_counter > MAX_SILENCE_FRAMES:
                    self._on_speech_ended()
            else:
                self.speech_buffer.extend(self.pre_speech_buffer)
                self.speech_buffer.append(indata_copy)
                self.silence_counter = 0
                self.pre_speech_buffer.clear()

    def _on_speech_started(self) -> None:
        """Handle speech onset.

        Passes the mutable ``speech_buffer`` reference directly (not a copied
        list) so that AudioProcessor can modify it in-place during processing.
        """
        self.speech_triggered = True
        logger.debug("Speech started")

        # Move pre-speech buffer into the output buffer and signal start
        self.speech_buffer.extend(self.pre_speech_buffer)
        self.on_chunk(self.speech_buffer)  # Pass mutable reference directly
        self.pre_speech_buffer.clear()
        self.silence_counter = 0

    def _on_speech_ended(self) -> None:
        """Handle speech offset.

        Appends ``None`` in-place to the same mutable list being iterated by
        the consumer thread, then resets ``speech_buffer`` for the next segment.
        """
        self._on_speech_end()
        self.speech_triggered = False
        logger.debug("Speech ended")

        # Append None in-place so consumer's while loop sees it and exits naturally.
        self.speech_buffer.append(None)
        self.speech_buffer = []  # Fresh list for next segment
        self.silence_counter = 0
