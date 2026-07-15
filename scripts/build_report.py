#!/usr/bin/env python
"""Build the report figures and manifest from existing run artifacts.

Reads artifacts without modifying them; performs no retraining,
recalibration, or benchmark recomputation. Fails loudly if a required
artifact or field is missing. Outputs are byte-deterministic; the
manifest records source paths, SHA-256 hashes, and field-level
provenance labels for every figure.

Usage:
    python scripts/build_report.py \
        [--validation-dir runs/validation-012] \
        [--summary runs/validation-012/observer-study-summary.json] \
        [--out docs/report/figures]
"""

from __future__ import annotations

import argparse
import sys

from quantscope.reporting import build_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", default="runs/validation-012")
    parser.add_argument("--summary", default="runs/validation-012/observer-study-summary.json")
    parser.add_argument("--out", default="docs/report/figures")
    args = parser.parse_args()
    manifest = build_report(args.validation_dir, args.summary, args.out)
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
