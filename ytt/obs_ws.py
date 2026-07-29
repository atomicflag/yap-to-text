"""OBS WebSocket client: sends closed captions to OBS Studio."""

import json
import logging

import websocket

logger = logging.getLogger(__name__)


class ObsWsClient:
    """Client for sending closed captions to OBS Studio via WebSocket.

    Connects to the OBS WebSocket server (default: ws://localhost:4444) and
    sends ``SendCaptions`` requests for each committed transcription segment.
    Runs its own event loop in a background thread via ``run()``.
    """

    def __init__(self, *, host: str = "localhost", port: int = 4444) -> None:
        """Initialize the OBS WebSocket client.

        Args:
            host: OBS WebSocket server hostname. Defaults to ``localhost``.
            port: OBS WebSocket server port. Defaults to ``4444``.
        """
        self._host: str = host
        self._port: int = port
        self._url: str = f"ws://{self._host}:{self._port}"
        self._ws: websocket.WebSocketApp | None = None

    def run(self) -> None:
        """Start the WebSocket connection event loop.

        Blocks until the connection is closed (e.g. via ``close()``).
        Call this in a background thread.
        """

        def _on_open(ws: websocket.WebSocketApp) -> None:
            logger.info("Connected to OBS WebSocket at %s", self._url)

        def _on_error(ws: websocket.WebSocketApp, error: Exception) -> None:
            logger.warning("OBS WebSocket error: %s", error)

        def _on_close(
            ws: websocket.WebSocketApp,
            close_status_code: int,
            close_msg: str | None,
        ) -> None:
            logger.info(
                "OBS WebSocket closed (status=%s, msg=%s)", close_status_code, close_msg
            )

        self._ws = websocket.WebSocketApp(
            self._url,
            on_open=_on_open,
            on_error=_on_error,
            on_close=_on_close,
        )

        self._ws.run_forever()

    def send_caption(self, text: str) -> None:
        """Send a closed caption to OBS Studio."""
        message: dict = {
            "request-type": "SendCaptions",
            "message-id": "",
            "text": text,
        }
        payload: str = json.dumps(message)

        if self._ws is not None:
            self._ws.send(payload)

    def close(self) -> None:
        """Close the WebSocket connection, if open."""
        if self._ws is not None:
            self._ws.close()
