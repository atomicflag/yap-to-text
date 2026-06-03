"""Twitch chat integration: OAuth authentication and message sending."""

import asyncio
import logging
import threading
from pathlib import Path

from twitchAPI.chat import Chat
from twitchAPI.oauth import UserAuthenticator, refresh_access_token
from twitchAPI.twitch import Twitch
from twitchAPI.type import AuthScope, ChatEvent

from ytt.config import TwitchConfig, load_config, save_config

logger = logging.getLogger(__name__)


class TwitchChatClient:
    """Manages a background Twitch chat connection."""

    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        refresh_token: str | None,
        target_channel: str,
        config_path: Path = Path("config.json"),
    ) -> None:
        """Initialize the Twitch chat client.

        Args:
            app_id: Twitch application ID.
            app_secret: Twitch application secret.
            refresh_token: OAuth refresh token (None for first-time interactive auth).
            target_channel: The Twitch channel to send messages to.
            config_path: Path to the config file for saving tokens.
        """
        self._app_id = app_id
        self._app_secret = app_secret
        self._refresh_token = refresh_token
        self._target_channel = target_channel
        self._config_path = config_path
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._ready: asyncio.Event = asyncio.Event()
        self._lock: threading.Lock = threading.Lock()

    def send_message(self, message: str) -> None:
        """Queue a message to be sent via Twitch chat. Thread-safe.

        Args:
            message: The text to send to the Twitch channel.

        """
        with self._lock:
            self._queue.put_nowait(message)

    async def _run_loop(self, stop: threading.Event) -> None:
        """Async event loop: authenticate, wait for ready, dispatch messages.

        Runs until ``stop`` is set (by the caller).
        """
        twitch = await Twitch(self._app_id, self._app_secret)

        if self._refresh_token is None:
            # First-time auth: no refresh token — interactive OAuth flow (like old code)
            logger.info("No refresh token — starting interactive OAuth...")
            auth = UserAuthenticator(
                twitch,
                [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT],
            )
            self._refresh_token, new_refresh_token = await auth.authenticate()
            # Save obtained refresh token to config for future runs
            config = load_config(self._config_path)
            if config.twitch is None:
                config.twitch = TwitchConfig(
                    app_id=self._app_id,
                    app_secret=self._app_secret,
                    target_channel=self._target_channel,
                )
            config.twitch.refresh_token = new_refresh_token
            save_config(config, self._config_path)
        else:
            # Normal flow: refresh existing token
            _, new_refresh_token = await refresh_access_token(
                self._refresh_token, self._app_id, self._app_secret
            )

        await twitch.set_user_authentication(
            None,  # type: ignore[arg-type] — token is set via refresh flow
            [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT],
            new_refresh_token,
        )

        chat = await Chat(twitch)

        async def _on_ready(event: ChatEvent) -> None:
            self._ready.set()

        chat.register_event(ChatEvent.READY, _on_ready)
        chat.start()

        try:
            while not self._ready.is_set():
                await asyncio.sleep(1)

            while True:
                while self._queue.empty():
                    await asyncio.sleep(1)
                    if stop.is_set():
                        raise KeyboardInterrupt from None

                with self._lock:
                    message = self._queue.get_nowait()

                await chat.send_message(self._target_channel, message)
        except (KeyboardInterrupt, Exception):
            pass
        finally:
            logger.info("Stopping Twitch thread")
            chat.stop()
            await twitch.close()

    def run(self, stop: threading.Event) -> None:
        """Blocking entry point — runs the Twitch chat event loop in this thread.

        Runs until ``stop`` is set. Follows the same pattern as
        BroadcastManager.run() (was osc_input_thread_loop).
        """
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._run_loop(stop))
        finally:
            loop.close()
