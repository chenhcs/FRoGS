"""Small I/O helpers shared by FRoGS scripts.

The main entry point is :func:`read_csv_auto`, a ``pandas.read_csv`` wrapper
that transparently handles ``.gz``-compressed counterparts of the requested
file.  This lets scripts keep their historical default paths
(``*.csv``) while the shipped data happens to be stored compressed
(``*.csv.gz``), and vice versa.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple

import pandas as pd


def _candidate_paths(path: str) -> List[str]:
    """Return an ordered list of paths to try for ``path``.

    Always starts with the verbatim path.  If the path ends with ``.gz`` we
    also try the uncompressed form; otherwise we additionally try the
    ``.gz``-suffixed form.  The ordering is chosen so that if the user
    *explicitly* asks for a compressed file we look there first.
    """
    candidates = [path]
    if path.endswith(".gz"):
        candidates.append(path[: -len(".gz")])
    else:
        candidates.append(path + ".gz")
    # Deduplicate while preserving order.
    seen = set()
    uniq: List[str] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def resolve_data_path(path: str) -> str:
    """Resolve ``path`` to the first existing candidate on disk.

    Tries ``path`` itself and a ``.gz`` counterpart.  Raises
    :class:`FileNotFoundError` with a clear, multi-line message listing every
    location that was probed.
    """
    tried: List[str] = []
    for candidate in _candidate_paths(path):
        tried.append(candidate)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not locate data file. Tried:\n  - " + "\n  - ".join(tried)
    )


def read_csv_auto(path: str, **read_csv_kwargs) -> pd.DataFrame:
    """Drop-in replacement for :func:`pandas.read_csv` with ``.gz`` fallback.

    Behaves identically to ``pd.read_csv`` for files that exist at the
    requested location.  If the file is missing we look for a ``.gz`` (or
    unsuffixed) sibling and, if found, read that instead so scripts work
    whether the data was pre-decompressed or not.  Pandas infers the
    compression from the suffix automatically.
    """
    resolved = resolve_data_path(path)
    return pd.read_csv(resolved, **read_csv_kwargs)


def validate_required_files(paths: Iterable[str]) -> Tuple[bool, List[str]]:
    """Check that each of ``paths`` exists (treating ``.gz`` alternates as OK).

    Returns a ``(ok, missing)`` tuple where ``missing`` is the list of inputs
    for which neither the path itself nor its ``.gz`` / un-``.gz`` counterpart
    could be found.  Useful for up-front validation in CLI scripts.
    """
    missing: List[str] = []
    for path in paths:
        try:
            resolve_data_path(path)
        except FileNotFoundError:
            missing.append(path)
    return (not missing), missing


def ensure_dir(path: Optional[str]) -> None:
    """Create ``path`` as a directory if it does not already exist."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)
