"""OBS WebSocket client: sends broadcast captions to OBS Studio (v5 protocol)."""

import logging

from obswebsocket import obsws, requests

logger = logging.getLogger(__name__)


class ObsWsClient:
    """Client for sending broadcast captions to OBS Studio (v5 protocol).

    Uses the obs-websocket-py library to handle the v5 protocol handshake
    (Hello → Identify → Identified) and SendStreamCaption requests.
    Sends captions to stream output so platforms like Twitch can display them.
    """

    def __init__(self, *, host: str = "localhost", port: int = 4444) -> None:
        """Initialize the OBS WebSocket client.

        Args:
            host: OBS WebSocket server hostname. Defaults to ``localhost``.
            port: OBS WebSocket server port. Defaults to ``4444``.
        """
        self._host: str = host
        self._port: int = port
        self._ws: obsws = obsws(host, port, "")
        self._ws.on_connect = self._on_connect
        self._ws.connect()

    @staticmethod
    def _on_connect(client: obsws) -> None:
        logger.info("Connected to OBS WebSocket at %s:%s", client.host, client.port)

    def send_caption(self, text: str) -> None:
        """Send a broadcast caption to OBS for stream output.

        This sends the caption to the stream output so platforms like
        Twitch can display it as "Closed Captions" for viewers.
        """
        self._ws.call(requests.SendStreamCaption(captionText=text))

    def close(self) -> None:
        """Close the WebSocket connection, if open."""
        self._ws.disconnect()
