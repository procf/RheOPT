# Analysis

Scripts behind the numbers quoted in the paper beyond the mean errors. They read run outputs
through `rheopt.runs` (`runs/`, or the original runs with `RHEOPT_LEGACY=<folder>`) and the data in `data/`, and write
CSV files to `results/` (or `--out`, `$RHEOPT_RESULTS`; not tracked).

| script | computes |
|---|---|
| `summarize_runs.py` | table of all `metrics.json` under a runs folder (setup, fluid, N, arm, seed, mean errors); use it for new runs and for the additional experiments (E2–E5) |
| `wake_frequency.py` | recovery of the shedding frequency per run and case; RMS of v_y along the wake for the Re = 94.1 case |
| `stress_components.py` | stress error split into shear stress, first normal-stress difference and mean normal stress; pooled errors |
| `persistence_baseline.py` | error of repeating the last input frame |
| `field_statistics.py` | mean and s.d. of every field per simulation (input window, predicted window, whole trajectory) |
| `viscosity_ratio.py` | solvent / total shear-stress ratio of the viscoelastic contraction data |
| `inference_timing.py` | inference time of a trained model |
