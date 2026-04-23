"""Smoke test for the modernized FRoGS utilities.

Exercises:
 * ``src/utils/parallel.py``: sequential + parallel paths of ``parallel.map``.
 * ``src/utils/io_utils.py``: transparent ``.gz`` fallback in ``read_csv_auto``.

Run it from the ``demo/`` directory::

    python smoke_test.py

No model weights or large datasets are required; we only read the first
few rows of the shipped L1000 file to prove the code path works.
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SRC = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC)

from utils import parallel  # noqa: E402  (import after sys.path tweak)
from utils.io_utils import read_csv_auto, resolve_data_path  # noqa: E402


def square(x: int) -> int:
    time.sleep(0.01)
    return x * x


def main() -> int:
    print("[1/3] Sequential parallel.map...")
    out = parallel.map(square, list(range(8)), n_CPU=1, progress=True)
    assert out == [x * x for x in range(8)], out
    print("   OK", out)

    print("[2/3] Parallel parallel.map (n_CPU=2)...")
    out = parallel.map(square, list(range(8)), n_CPU=2, progress=True)
    assert out == [x * x for x in range(8)], out
    print("   OK", out)

    print("[3/3] read_csv_auto with transparent .gz fallback...")
    # The shipped repo has `L1000_PhaseI_and_II.csv.gz` but scripts default
    # to the `.csv` name.  resolve_data_path should find the .gz sibling.
    requested = os.path.join(REPO_ROOT, "data", "L1000_PhaseI_and_II.csv")
    resolved = resolve_data_path(requested)
    print(f"   resolved {requested!r} -> {resolved!r}")
    df = read_csv_auto(requested, nrows=3)
    print("   OK, head:")
    print(df.head(3).to_string(index=False))

    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
