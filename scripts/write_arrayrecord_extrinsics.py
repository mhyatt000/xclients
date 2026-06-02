from __future__ import annotations

from pathlib import Path
import sys

import tyro

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from xclients.dream_dr.arrayrecord_extrinsics import main, MaterializeConfig


def cli() -> None:
    main(tyro.cli(MaterializeConfig))


if __name__ == "__main__":
    cli()
