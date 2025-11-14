#!/usr/bin/env python
"""Small helper script to visualize a cProfile .prof file using snakeviz.

Usage (from project root):

    uv run python scripts/view_profile.py \
        --profile profiling_results/rebalance_profile.prof

This will launch snakeviz in your browser for interactive exploration
of the profiling data.
"""

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a cProfile .prof file using snakeviz",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="profiling_results/rebalance_profile.prof",
        help="Path to the .prof file produced by the profiling script",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_path = Path(args.profile).expanduser().resolve()

    if not profile_path.exists():
        raise SystemExit(f"Profile file not found: {profile_path}")

    # Delegate to snakeviz CLI; relies on snakeviz being installed in the env.
    try:
        subprocess.run(["snakeviz", str(profile_path)], check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            "snakeviz command not found. Make sure it is installed in your uv environment:\n"
            "    uv pip install snakeviz"
        ) from exc


if __name__ == "__main__":
    main()
