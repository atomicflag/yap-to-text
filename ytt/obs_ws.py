"""OBS WebSocket client: sends closed captions to OBS Studio."""

import contextlib
import json
import logging
import threading

from websocket import WebSocketException, create_connection

logger = logging.getLogger(__name__)

_MAX_RETRIES: int = 3


class ObsWsClient:
    """Client for sending closed captions to OBS Studio via WebSocket.

    Connects to the OBS WebSocket server (default: ws://localhost:4444) and
    sends ``SendCaptions`` requests for each committed transcription segment.
    Connection is lazy — the first call to ``send_caption`` will attempt to
    connect. After 3 failed attempts the client permanently gives up and
    silently ignores further send attempts.
    """

    def __init__(self, *, host: str = "localhost", port: int = 4444) -> None:
        """Initialize the OBS WebSocket client.

        Args:
            host: OBS WebSocket server hostname. Defaults to ``localhost``.
            port: OBS WebSocket server port. Defaults to ``4444``.
        """
        self._host: str = host
        self._port: int = port
        self._conn = None  # type: ignore[assignment]
        self._retry_count: int = 0
        self._giving_up: bool = False
        self._lock: threading.Lock = threading.Lock()

    def send_caption(self, text: str) -> None:
        """Send a closed caption to OBS Studio.

        If the connection is not yet established, it will be opened lazily.
        After ``_MAX_RETRIES`` failed connection attempts the client
        permanently gives up and silently ignores further calls.
        """
        if self._giving_up:
            return

        message: dict = {
            "request-type": "SendCaptions",
            "message-id": "",
            "text": text,
        }
        payload: str = json.dumps(message)

        with self._lock:
            should_reconnect = False
            if self._conn is None:
                should_reconnect = True
            else:
                try:
                    self._conn.send(payload)
                    return
                except WebSocketException:
                    logger.warning(
                        "OBS WebSocket send failed — closing stale connection"
                    )
                    with contextlib.suppress(WebSocketException):
                        self._conn.close()
                    self._conn = None
                    should_reconnect = True

            if should_reconnect:
                self._retry_count += 1
                try:
                    url = f"ws://{self._host}:{self._port}"
                    self._conn = create_connection(url, timeout=5)
                    self._conn.send(payload)
                    logger.info("Connected to OBS WebSocket at %s", url)
                except (WebSocketException, OSError) as exc:
                    logger.warning(
                        "OBS WebSocket connection failed (%d/%d): %s",
                        self._retry_count,
                        _MAX_RETRIES,
                        exc,
                    )
                    self._conn = None
                    if self._retry_count >= _MAX_RETRIES:
                        logger.warning(
                            "OBS WebSocket giving up after %d failed attempts",
                            _MAX_RETRIES,
                        )
                        self._giving_up = True

    def close(self) -> None:
        """Close the WebSocket connection, if open."""
        with self._lock:
            if self._conn is not None:
                with contextlib.suppress(WebSocketException):
                    self._conn.close()
                self._conn = None
