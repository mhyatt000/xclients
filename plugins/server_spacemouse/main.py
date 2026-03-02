"""CLI client for interacting with the SpaceMouse webpolicy server."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from rich import print

# Allow importing local viewer implementation without installing the package
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from server_spacemouse.viewer import Viewer
from webpolicy.client import Client


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--interval", type=float, default=0.1, help="Polling interval in seconds")
    parser.add_argument("--count", type=int, default=0, help="Number of polls before exiting (0 = infinite)")
    parser.add_argument("--viewer", action="store_true", help="Display the viewer window for each poll")
    parser.add_argument("--viewer-window", default="SpaceMouse Viewer", help="Window title when rendering")
    return parser.parse_args(argv)


def poll_loop(
    client: Client,
    interval: float,
    count: int,
    viewer: Viewer | None,
    cv2_mod,
    window_title: str,
) -> None:
    remaining = count
    while remaining == 0 or remaining > 0:
        payload = client.step({})
        print(payload)
        if viewer is not None and cv2_mod is not None:
            vector = payload.get("vector")
            if vector is None or len(vector) < 6:
                print("[red]Viewer: missing vector data[/red]")
            else:
                image = viewer.step(np.asarray(vector, dtype=float))
                bgr = image[..., ::-1]
                cv2_mod.imshow(window_title, bgr)
                if cv2_mod.waitKey(1) & 0xFF == ord("q"):
                    break
        if remaining == 0:
            time.sleep(interval)
            continue
        remaining -= 1
        if remaining > 0:
            time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    client = Client(host=args.host, port=args.port)
    viewer = None
    cv2_mod = None
    if args.viewer:
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            print(f"[red]Viewer mode requires opencv-python: {exc}[/red]")
            return 1
        cv2_mod = cv2
        viewer = Viewer()

    try:
        poll_loop(client, args.interval, args.count, viewer, cv2_mod, args.viewer_window)
    except KeyboardInterrupt:
        print("[yellow]\nInterrupted by user[/yellow]")
    finally:
        if cv2_mod is not None:
            cv2_mod.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
