"""CLI client for interacting with the SpaceMouse webpolicy server."""

from __future__ import annotations
import jax

import argparse
import json
import sys
import time
from rich import print

from webpolicy.client import Client


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--interval", type=float, default=0.1, help="Polling interval in seconds")
    parser.add_argument("--count", type=int, default=0, help="Number of polls before exiting (0 = infinite)")
    return parser.parse_args(argv)


def poll_loop(client: Client, interval: float, count: int) -> None:
    remaining = count

def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    client = Client(host=args.host, port=args.port)

    while True:
        payload = client.step({})
        # payload = jax.tree.map(lambda x: round(x, 3) if isinstance(x, float) else x, payload)
        print(payload)


if __name__ == "__main__":
    main()
