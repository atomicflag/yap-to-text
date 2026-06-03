"""Dispatches transcription results to OSC, Twitch, clipboard, and console.

BroadcastManager coordinates output to all configured channels based on the
transcription buffer state.
"""

import logging
import threading
import time

import numpy as np
import pyperclip
import sounddevice as sd
import soundfile as sf
from pythonosc import udp_client

from stt.transcription import TranscriptionBuffer
from stt.twitch import TwitchChatClient

logger = logging.getLogger(__name__)

# Audio output constants
BLIP_FILE: str = "blip.wav"

VRC_TEXTBOX_INTERVAL: float = 1.5


class BroadcastManager:
    """Manages all transcription output channels.

    Routes committed and intermediate transcriptions to VRC OSC, Twitch chat,
    system clipboard, or OBS browser source depending on the active mode.
    """

    def __init__(
        self,
        *,
        transcription_buffer: TranscriptionBuffer,
        twitch_mode: bool = False,
        clipboard_mode: bool = False,
        twitch_chat: TwitchChatClient | None = None,
    ) -> None:
        """Initialize the broadcast manager.

        Args:
            transcription_buffer: Buffer to read transcription text from.
            twitch_mode: If True, send text to Twitch chat instead of VRC OSC.
            clipboard_mode: If True, copy completed transcriptions to system clipboard.
            twitch_chat: Optional TwitchChatClient for sending messages to Twitch.
                         Provided when twitch_mode is True.

        """
        self.transcription_buffer: TranscriptionBuffer = transcription_buffer
        self.twitch_mode: bool = twitch_mode
        self.clipboard_mode: bool = clipboard_mode
        self._twitch_chat: TwitchChatClient | None = twitch_chat
        self.speech_triggered: bool = False

        # OSC client for VRC (non-Twitch mode)
        self._osc: udp_client.SimpleUDPClient | None = None
        if not twitch_mode:
            self._osc = udp_client.SimpleUDPClient("127.0.0.1", 9000)

        # Text buffer state for OSC input (non-Twitch mode)
        self._osc_text: str | None = None

        # Load blip audio data
        try:
            blip_data, blip_samplerate = sf.read(BLIP_FILE, dtype="float32")
            self._blip_data: np.ndarray = blip_data
            self._blip_samplerate: int = blip_samplerate
        except OSError:
            logger.exception("Can't load blip.wav")
            raise

    # Public interface

    def on_commit(self) -> None:
        """Handle a committed transcription segment.

        Reads from the transcription buffer and broadcasts based on mode:

        - Twitch: sends only the last committed segment
        - OSC: sets truncated running text for VRC chatbox
        - Clipboard: copies full untruncated text to system clipboard
        """
        # Clipboard copy (always active if mode enabled, uses full text)
        if self.clipboard_mode:
            pyperclip.copy(self.transcription_buffer.get_full())

        # Twitch mode → send last segment only to chat via client
        if self.twitch_mode and self._twitch_chat is not None:
            text = self.transcription_buffer.last()
            if text is not None:
                self._twitch_chat.send_message(text)
            return

        # Non-Twitch: set OSC input text for the input thread to dispatch
        self._osc_text = self.transcription_buffer.get()

    def on_intermediate(self) -> None:
        """Handle an intermediate transcription update.

        Reads from the transcription buffer for real-time display.
        """
        if self.twitch_mode:
            return  # Twitch chat does not show intermediate updates

        if self._osc:
            self._osc_text = self.transcription_buffer.get()

    def on_speech_start(self) -> None:
        """Notify broadcast manager that speech has started.

        Sends typing indicator and shows current buffer state with ellipsis.
        """
        self.speech_triggered = True
        if self.twitch_mode:
            return

        if self._osc:
            self._osc.send_message("/chatbox/typing", True)
            self._osc_text = self.transcription_buffer.get(True)

    def on_speech_end(self) -> None:
        """Handle end of a speech segment.

        Plays a blip sound (if available) and stops the typing indicator.
        """
        self.speech_triggered = False
        if self.twitch_mode:
            try:
                sd.play(self._blip_data, self._blip_samplerate)
            except OSError as exc:
                logger.warning("Failed to play blip sound: %s", exc)

        if self._osc:
            self._osc.send_message("/chatbox/typing", False)

    def on_erase(self) -> None:
        """Clear the OSC text when an erase keyword is detected."""
        if self._osc:
            self._osc_text = ""

    # OSC input thread

    def run(self, stop: threading.Event) -> None:
        """Thread loop that periodically dispatches OSC /chatbox/input messages.

        Runs until `stop` is set. Same pattern as TwitchChatClient.run().

        Args:
            stop: threading.Event that is set to signal shutdown.

        """
        while not stop.is_set():
            if self._osc_text is not None:
                if self._osc:
                    self._osc.send_message("/chatbox/input", [self._osc_text, True])
                self._osc_text = None
            time.sleep(VRC_TEXTBOX_INTERVAL)
