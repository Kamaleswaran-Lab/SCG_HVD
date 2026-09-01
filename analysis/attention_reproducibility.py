# Checks whether the attention result holds across seeds, before it is reported.
"""The first attention figure came from one trained fold. That describes the fold; it does not
establish anything about the architecture. The criterion fixed in
BPEX_R1/interpretability-plan.md is that the axis result has to reproduce in all three seeds.

The quantity checked is how much each axis's attention varies over the cardiac cycle, as a
fraction of its own mean. An axis the model reads selectively varies; an axis it ignores sits
flat near uniform. The claim in the manuscript is that z dominates, so z's variation must
exceed both x and y in every seed.

Usage.
    python analysis/attention_reproducibility.py --out out/interp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def variation(df: pd.DataFrame) -> dict:
    """Coefficient of variation of the beat-aligned attention, per axis."""
    return {ax: float(g.attention.std() / g.attention.mean())
            for ax, g in df.groupby("axis")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("out/interp"))
    ap.add_argument("--task", default="task1")
    ap.add_argument("--model", default="fusion")
    ap.add_argument("--out", type=Path, default=Path("out/interp"))
    a = ap.parse_args()

    rows, per_seed = [], {}
    for d in sorted(a.root.glob("seed*")):
        f = d / f"{a.task}_{a.model}_beat_aligned_attention.csv"
        if not f.exists():
            continue
        v = variation(pd.read_csv(f))
        per_seed[d.name] = v
        rows.append({"seed": d.name, **{f"cv_{k}": round(x, 4) for k, x in v.items()}})
    if not rows:
        print(f"no per-seed results under {a.root}"); return

    t = pd.DataFrame(rows)
    print("Coefficient of variation of attention over the cardiac cycle, by axis")
    print(t.to_string(index=False))

    holds = {s: (v.get("z", 0) > v.get("x", 0) and v.get("z", 0) > v.get("y", 0))
             for s, v in per_seed.items()}
    n_ok = sum(holds.values())
    verdict = {"per_seed": per_seed, "z_dominant": holds,
               "n_seeds": len(holds), "n_holding": n_ok,
               "reportable": bool(len(holds) >= 3 and n_ok == len(holds))}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.task}_{a.model}_attention_reproducibility.json").write_text(
        json.dumps(verdict, indent=2))

    print(f"\nz dominates both other axes in {n_ok} of {len(holds)} seeds")
    print("verdict:", "reproducible across seeds; reportable."
          if verdict["reportable"] else
          "does not hold in every seed; report as a single-fold observation or not at all.")


if __name__ == "__main__":
    main()
