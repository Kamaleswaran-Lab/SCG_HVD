# SCG_HVD

Code for *Deep Learning and Dual-Domain Fusion of Temporal and Spectrotemporal SCG for
Valvular Heart Disease Classification*.

The models classify valvular heart disease from tri-axial seismocardiogram recordings by
combining a temporal encoder over the waveform with an image encoder over continuous wavelet
transform scalograms of the same signal.

## What this repository is for

It reproduces every number in the paper, and it contains the analyses that went into the
revision — including the ones whose results are not in the paper. If you want to check whether
a reported figure holds, the code that produced it is here, and so is the code that produced
the figures we chose not to report.

Two things are worth knowing before reading any result.

**The headline number depends entirely on how the data is split.** Under the segment-level
split used in earlier work on these datasets, the fused model reaches 99.63% accuracy on Task
I. Under a patient-level split it reaches 65.3%. Both are in the paper. The first is not a
generalization estimate: windows are 10 s long on a 5 s stride, so 98.96% of test segments
share five seconds of raw signal with a training segment, and no patient is held out at all.
`scg_hvd/splits.quantify_overlap_leakage()` measures this so you do not have to take our word
for it.

**The claim is narrow.** It is that adding a spectrotemporal branch to a temporal SCG encoder
improves it — not that this architecture beats other sequence models. The ablation that tests
it holds the temporal branch fixed and varies the image backbone.

## Layout

    scg_hvd/          models, datasets, splits, training loop, metrics
    scripts/          entry points and SLURM submission scripts
    analysis/         the analyses and the figures they produce
    archive/          verbatim copies of the scripts that produced the published numbers

`archive/` is evidence rather than code: those files are kept byte for byte as found, with
their md5 sums recorded in `archive/README.md`, which is also why they are the only files here
with Korean comments.

## Setup

    pip install -r requirements.txt
    export SCG_HVD_DATA=/path/to/prepared/data

`SCG_HVD_DATA` should hold:

    Task1/  Task2/                 10 s windows as .npy, one subdirectory per class
    Task1_images/  Task2_images/   per-axis CWT scalograms as .png
    meta/                          df_metadata.csv and the segment index files

The index files carry the absolute paths of the machine that built them;
`scg_hvd.paths.localise` rewrites them onto your root by matching the `Task1/` or `Task2/`
component, so they do not need editing.

The datasets themselves are not redistributed here. They come from
[Yang et al.](https://doi.org/10.1038/s41597-021-01046-y) and
[Kaisti et al.](https://doi.org/10.1109/JBHI.2018.2872984) under their own terms.

## Running things

Patient-level cross-validation, which is the primary evaluation:

    python scripts/run_patient_cv.py --task task1 --model fusion --folds 5 --seeds 0 1 2

`--model` takes any name in `scg_hvd.models.MODELS`: `1d`, `2d`, `fusion`, the backbone
variants such as `fusion_resnet18`, and the capacity-matched baselines.

Reproducing the published segment-level numbers and diffing them against the manuscript:

    python scripts/run_paper_segment.py --task task1 --models 1d 2d fusion --out out/paper
    python analysis/compare_to_manuscript.py

Collecting results and building the figures:

    python analysis/final_report.py --task task1
    python analysis/make_figures.py --out out/figures

The `scripts/submit_*.sh` files are SLURM array jobs for the same entry points. Their
partition and QoS lines are specific to the cluster this ran on; change those two lines and
they should work elsewhere. Repository root and interpreter come from `SCG_HVD_REPO` and
`SCG_HVD_PYTHON`, both of which have sensible defaults.

## The analyses behind the revision

| Script | What it answers |
|---|---|
| `analysis/final_report.py` | Patient-level results, both baselines, paired tests |
| `analysis/compare_overlap.py` | Whether removing window overlap costs performance |
| `analysis/covariate_report.py` | How much age, sex and heart rate alone explain |
| `analysis/extract_heart_rate.py` | Heart rate, from the ECG inside the segment files |
| `analysis/cohort_table.py` | The cohort table and per-patient segment counts |
| `analysis/axis_alignment.py` | Whether the two datasets label their axes alike |
| `analysis/interpretability.py` | Attention weights over the cardiac cycle, Grad-CAM |
| `analysis/compare_to_manuscript.py` | Reproduction against all 120 published cells |

## Things we found and did not hide

The audit that preceded this revision turned up several errors in our own published
description. They are corrected in the revised manuscript and documented in the code:

- The paper described three independent image backbones. The code that produced every reported
  number uses one backbone shared across the three axes, which is a third of the parameters.
  See `scg_hvd/models.py`.
- The paper described a complex Morlet wavelet over 1–30 Hz, power scalograms and an RGB
  colormap. The implementation uses a real Morlet wavelet over scales 1–128 (1.63–208 Hz),
  magnitude, and grayscale.
- The archived pipeline recovered metrics by multiplying normalised confusion matrices back
  through a hard-coded support. Seventeen of the 120 published cells disagree in the second
  decimal place as a result. `scg_hvd/metrics.py` computes everything from raw predictions.
- The archived LOOCV harness chose validation patients using the test patient's label.
  `scg_hvd/splits.py` does not.

## Citing

Please cite the paper. A `CITATION.cff` will be added once the DOI is assigned.

## License

MIT, see `LICENSE`. The datasets are not covered by it.
