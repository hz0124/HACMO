#!/usr/bin/env python3
"""CLI for SupernodeSim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import simulate_layer
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="SupernodeSim simulator")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sim = sub.add_parser("simulate", help="simulate one layer workload")
    sim.add_argument("--config", required=True)
    sim.add_argument("--output", default="")
    args = parser.parse_args()

    if args.cmd == "simulate":
        config = load_config(args.config)
        result = simulate_layer(config).to_dict()
        text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
        print(text, end="")


if __name__ == "__main__":
    main()
