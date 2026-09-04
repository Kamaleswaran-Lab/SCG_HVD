# Preparing the data

`SCG_HVD_DATA` points at a directory that this repository reads but does not build. The two
public releases have to be turned into fixed-length segments and scalogram images first, in
two steps:

| Step | What it does |
|---|---|
| Segmentation | Cuts the resampled recordings into 10 s windows on a 5 s stride and writes the segment index |
| Scalograms | Renders each segment's three axes as grayscale CWT scalograms |

What matters is the result, not how it is produced, so this page states the layout the code
expects and the details that are easy to get wrong.

## What the result has to look like

    $SCG_HVD_DATA/
      Task1/<class>/    <patient>_seg<NNN>.npy           (T, C) float arrays
      Task2/<class>/
      Task1_images/     <patient>_seg<NNN>_{x,y,z}.png   224x224 grayscale
      Task2_images/
      meta/
        df_metadata.csv                cohort metadata for Dataset I
        segment_metadata_task1.csv     segment_id, patient_id, label, task, filepath,
        segment_metadata_task2.csv     start_time_sec

The `filepath` column holds absolute paths from wherever the index was built.
`scg_hvd.paths.localize` maps them onto the local root by keeping everything from the `Task1/`
or `Task2/` component onwards, so the index files are portable as they are.

## Details that matter

**Channel layout differs between the two datasets** and is inferred from the column count, not
from a flag. Dataset I files have 12 columns with SCG in 0-2; Dataset II files have 7 with SCG
in 1-3. See `scg_hvd/datasets.select_scg_channels`. Getting this wrong reads gyroscope or ECG
channels as accelerometer ones and produces plausible-looking nonsense.

**The scalograms are grayscale**, rendered from `pywt.cwt(signal, np.arange(1, 129), 'morl')`
as magnitude with no logarithmic compression, autoscaled per image. They are replicated to
three channels at load time. A colormapped variant exists in the original tree but was not used
for any published result.

**`start_time_sec` has to be exact.** The overlap diagnostics identify neighboring windows by
comparing start times to the stride, so an approximate value silently reports no leakage where
there is plenty.
