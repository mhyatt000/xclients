from __future__ import annotations

from pathlib import Path
import sys

import tyro


def cli() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from xclients.dream_dr.pipeline import Config, main

    main(tyro.cli(Config))


if __name__ == "__main__":
    cli()
