# Upstream node count and station-level relative peak response time

Analysis code for the manuscript *Upstream node count as a graph-derived proxy for station-level
relative peak response time in an urban wastewater network* (Urban Water Journal, submitted).

The outcome is the event-residualised peak response time of a sewer level gauge, `s_i` in hours
relative to the event mean. The predictor is `log(1 + up_size)`, the number of nodes from which flow
can reach the gauge's bound node in the flow-directed asset graph. Everything here serves the
decomposition the paper reports: whether that scalar orders response timing (H1), whether it is a
recoding of three computable hydraulic-scale indicators (H2), whether it adds to them out of fold
(H3), and whether it matches them at a lower data requirement (H4).

## Data

The asset files and telemetry are openly archived, and are **not** redistributed here:

> Sweetapple, C., T. Illingworth, and P. Melville-Shreeve. 2026. 'Dataset: High Resolution
> Hydrometric and Sewer System Monitoring of an Urban Catchment with Complex Flood Risk.'
> Zenodo dataset, version 1.0. https://doi.org/10.5281/zenodo.20699634

Set `SEWER_DATASET_ROOT` to the directory holding that dataset (it should contain
`index_telemetry_data.csv` and the per-gauge telemetry files) before running
`src/01_extract_station_event_peaks.py`. Every later script runs from the derived tables in
`derived/`, so the full pipeline can be re-run without the raw telemetry, and the scripts that
produce the paper's numbers can be re-run in seconds.

```
export SEWER_TIMING_DATA=/path/to/derived      # defaults to ../derived
export SEWER_DATASET_ROOT=/path/to/dataset     # only needed for step 01
```

## Layout

```
src/        analysis scripts, in pipeline order
derived/    station-level and fold-level tables the scripts read
results/    output tables as reported in the paper
```

## Pipeline

| Step | Script | Produces |
|---|---|---|
| 01 | `01_extract_station_event_peaks.py` | station x event time-to-peak table from raw telemetry (95 events, 92 with usable coverage) |
| 02 | `02_build_analysis_table.py` | `p1_h3_master.csv`: outcome, log count, hydraulic-scale indicators, fold assignments |
| 03 | `03_conditional_increment.py` | in-sample nested models and conditional increments (Table 2, panel A) |
| 04 | `04_block_cv_fold_consistent.py` | four blockings, fold-consistent outcome, pooled out-of-fold R2 (Table 2, panels B and C; Table 3, panel A) |
| 05 | `05_jackknife_intervals.py` | leave-one-fold-out jackknife intervals on pooled sums of squares (Table 2, panel D; note iii) |
| 06 | `06_permutation_test.py` | 999 block-preserving permutations of the count labels (Table 2, note iv) |
| 07 | `07_paired_bootstrap_naive_baselines.py` | paired station bootstrap, naive-baseline ladder (Table S13, Table S2) |
| 08 | `08_calibration_curve.py` | local intercept recalibration against the number of local gauges, m in {0, 5, 10, 20, 40, all} (Table 3, panel B) |
| 09 | `09_calibration_grid.py` | the same curve over all 20 partition configurations |
| 10 | `10_excluded_representativeness.py` | composition of the excluded gauges and the post-stratified slope (Table S16) |
| 11 | `11_competing_scale_variables.py` | upstream link count and upstream total conduit length at the same folds and target as the count (Table S18) |
| 12 | `12_slope_floor_sensitivity.py` | the slope floor for non-positive gradients: four floors and outright exclusion (Table S19) |

Steps 02 and 04 also write the fold structure used everywhere else, which is why later steps read
`derived/p4_cv_foldref*.csv` rather than recomputing a partition.

Both new steps re-derive the partition inside the script with the same call as step 04
(`kmeans2(..., minit="points", seed=0)` on bound-node coordinates), so that the k-means blocking is
identical to the one in Table 2; the count row of step 11 reproduces Table 2, panel C to four
decimals, which is the check that the partition and the fold-consistent outcome are the same.

## Notes on the estimators

- **Fold-consistent outcome.** `s_i` is defined against an event reference, so within each fold the
  reference is re-estimated from training-fold stations only and applied to both training and
  held-out outcomes (step 04). Training stations use the self-excluded form; held-out stations use
  the full training-fold mean.
- **Blockings.** 26 maximal upstream sets, 154 trunk segments, and coordinate k-means partitions at
  k = 27 and k = 10. The two topological blockings are strongly unbalanced, which is intrinsic to a
  converging network and is why the H3 verdict rests on the fold-level interval rather than on the
  trunk-segment point estimate.
- **Conditional increments.** Reported as `dR2 = R2(A u B) - R2(B)` on pooled out-of-fold sums of
  squares, not as the difference between two single-block columns.
- **Calibration curve.** Slope fixed at the training value, intercept estimated from m stations drawn
  within the held-out subregion, evaluated on the remainder (at least 8 evaluation stations per
  fold), 200 redraws. `m = 0` reproduces the zero-label pooled metric and `m = all` reproduces the
  full-subregion recalibration, which is the check that this implementation matches the paper
  (9.06 h and 8.45 h against the reported 9.1 h and 8.5 h).

## Requirements

Python 3.10+, `numpy`, `pandas`, `scipy`, `networkx`, `pyproj` (step 01 only). No compiled
extensions beyond the standard scientific stack; all regressions are solved in closed form so that
the fits are bit-reproducible.

## Coverage and known gaps

Steps 01-12 reproduce the main models, the block cross-validation, the jackknife and permutation
routines, the paired bootstrap, the calibration curve, the two competing size variables and the
slope-floor sensitivity. Two analyses reported in the paper are not in this archive: the ridge
comparison now in Supplementary Material Table S17 (the original Table 2, panel E), and the DEM-based
topographic check of Supplementary Material S7 (which depends on the 5 m elevation and infiltration
rasters of the same Zenodo record). Their numbers are reported in full in the manuscript and its
supplement.

## License

MIT for the code (see `LICENSE`). The dataset remains under the licence of its Zenodo record.
