# Resolves the data root, so no script has to hard-code where the segments live.
"""The datasets are not redistributed with this code and are not small, so their location is
configuration rather than something to commit.

Set `SCG_HVD_DATA` to the directory holding the prepared segments and scalograms:

    Task1/  Task2/                 10 s windows as .npy, one subdirectory per class
    Task1_images/  Task2_images/   per-axis CWT scalograms as .png
    meta/                          df_metadata.csv and the segment index files

`scripts/prepare_data.md` describes how these are built from the two public releases.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "SCG_HVD_DATA"


def data_root(required=True) -> Path:
    """The dataset root, from $SCG_HVD_DATA.

    Fails loudly rather than falling back to a path that happens to exist on one machine: a
    silently wrong data root produces results that look fine and are not.
    """
    raw = os.environ.get(ENV_VAR)
    if not raw:
        if not required:
            return Path()
        raise RuntimeError(
            f"{ENV_VAR} is not set. Point it at the directory holding Task1/, "
            f"Task1_images/ and meta/, e.g.\n"
            f"    export {ENV_VAR}=/path/to/scg_hvd_data"
        )
    p = Path(raw).expanduser()
    if required and not p.is_dir():
        raise RuntimeError(f"{ENV_VAR} points at {p}, which is not a directory.")
    return p


def localise(paths, root=None):
    """Rewrite the absolute paths in the segment index onto the local data root.

    The index files ship with the absolute paths of the machine that built them. Rather than
    substituting one hard-coded prefix for another, this keeps the part of each path from
    `Task1/` or `Task2/` onwards and hangs it off the local root, so an index built anywhere
    resolves here.
    """
    root = Path(root) if root is not None else data_root()

    def one(p: str) -> str:
        parts = Path(p).parts
        for i, part in enumerate(parts):
            if part in ("Task1", "Task2"):
                return str(root.joinpath(*parts[i:]))
        # No task directory in the path: assume it is already relative to the root.
        return str(root / Path(p).name)

    return paths.map(one)
