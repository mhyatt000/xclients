from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server
import logging
import pyspacemouse
import threading


logger = logging.getLogger(__name__)


class SpaceMousePolicy(BasePolicy):
    """Stream SpaceMouse readings to webpolicy clients."""

    def __init__(self) -> None:
        logger.info("Initializing SpaceMousePolicy")
        self._state = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        }
        self._lock = threading.Lock()
        self._running = True
        try:
            self._device_ctx = pyspacemouse.open()
            self._device = self._device_ctx.__enter__()
            logger.info("SpaceMouse device opened")
        except RuntimeError as exc:
            logger.exception("Failed to open SpaceMouse device")
            raise exc
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        logger.debug("Reader thread started")

    def _reader_loop(self) -> None:
        logger.debug("Reader loop running")
        while self._running:
            state = self._device.read()
            with self._lock:
                self._state.update(
                    x=state.x,
                    y=state.y,
                    z=state.z,
                    roll=state.roll,
                    pitch=state.pitch,
                    yaw=state.yaw,
                )

    def reset(self, *args, **kwargs) -> None:
        with self._lock:
            for key in self._state:
                self._state[key] = 0.0

    def step(self, obs: dict) -> dict:
        with self._lock:
            payload = dict(self._state)
        payload = {k: round(v, 3) if isinstance(v, float) else v for k, v in payload.items()}
        payload["vector"] = [
            payload["x"],
            payload["y"],
            payload["z"],
            payload["roll"],
            payload["pitch"],
            payload["yaw"],
        ]
        return payload

    def close(self) -> None:
        if not self._running:
            return
        self._running = False
        self._reader_thread.join(timeout=1)
        self._device_ctx.__exit__(None, None, None)
        logger.info("SpaceMousePolicy closed")

    def __del__(self) -> None:
        self.close()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start a webpolicy server that streams SpaceMouse events."""

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting SpaceMouse server on %s:%s", host, port)
    policy = SpaceMousePolicy()
    server = Server(policy=policy, host=host, port=port)
    try:
        server.serve()
    finally:
        policy.close()
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    main()
