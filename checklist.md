# Public release checklist

Goal: `SCG_HVD` can be handed to a reviewer or a stranger and read without us in the room.

## Comments and docstrings to English

- [x] `scg_hvd/`  -  datasets, metrics, models, splits, train
- [x] `analysis/`  -  10 files
- [x] `scripts/`  -  3 Python, 7 SLURM
- [x] `archive/README.md`
- [x] `archive/_canonical/*.py`  -  **deliberately left as found.** These are verbatim copies
      whose md5 sums are recorded; editing a comment would break that provenance.

## Portability

- [x] Data root from `$SCG_HVD_DATA`, with a clear error when unset
- [x] SLURM scripts: repo root, interpreter and partition from variables at the top
- [x] `requirements.txt` with the versions actually used

## Repository furniture

- [x] `README.md`  -  what this is, how to run it, what each result corresponds to
- [x] `LICENSE`
- [x] Drop the two empty directories, or give them a reason to exist

## Verify

- [x] `python -c "import scg_hvd"` from a clean checkout
- [x] Every script's `--help` runs
- [x] `grep -rP '[\x{AC00}-\x{D7A3}]'` finds nothing outside `archive/_canonical/`
