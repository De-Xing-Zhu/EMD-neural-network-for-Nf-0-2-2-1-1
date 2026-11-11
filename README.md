# EMD-neural-network-for-Nf-0-2-2-1-1
NN training code for the Nf=0,2,2+1+1 flavors from LQCD data for holographic EMD model
# EMD Neural-Network Training Package — Nf = 0, 2, 2+1+1

Training programs for the machine-learning reconstruction of 5D
Einstein–Maxwell–dilaton (EMD) backgrounds from lattice-QCD thermodynamics,
for three flavor sectors:

- `nf0/`  — pure gauge (Nf = 0), entropy-only fit
- `nf2/`  — Nf = 2, lattice-only joint fit (spline gauge function)
- `nf211/` — Nf = 2+1+1, lattice-only fit and the controlled HRG comparison

This package contains **code only, no data**. Place the input data files
listed below into each `data/` directory and run the commands from this
package root. All scripts locate their `data/` and `results/` directories
relative to their own location, so the package can be moved anywhere.

## Dependencies

Python 3 with `numpy`, `scipy`, `torch`, `matplotlib`.
The GPU training scripts (`--device cuda`) need a CUDA-enabled PyTorch;
edit the `--device` argument to `cpu` for CPU runs (much slower).

## Required input data (NOT included)

### nf0/data/
- `Nf0_train.csv` — 16 pure-gauge lattice points, columns `T[GeV], s_over_T3, sigma`
  (deconfined s/T^3, Borsanyi et al. 2012)
- `pure_gauge_Borsanyi2012.csv` — optional full lattice table, only used for
  plotting comparisons

### nf2/data/
- `Nf2_train_s_verified.csv` — Nf=2 entropy rows from the Burger et al. EOS,
  columns `T, s_over_T3, sigma, kind`
- `Nf2_train_chi_verified.csv` — chi_2^B rows (Datta et al.), `T, chi2B, sigma, kind`
- `Nf2_train_chi4_verified.csv` — chi_4^B rows from the same dataset
- `Nf2_nB_muhat1_verified.csv` — Taylor-derived n_B/T^3 target at mu_B/T = 1
- `newSEXP_eos.csv` — source EOS table (used by target rebuild / evaluation)
- `source/Fig4b.eps` — only needed to rebuild the verified targets with
  `build_lqcd_targets.py` (EPS error-bar parsing)
- `hrg_adapted_Burger2015.csv` — only needed by the legacy HRG/bridge
  variants (`train_nf2_hrg140_gpu.py`, `dense_low_temperature_eos` paths);
  the LQCD-only fit does not use it

### nf211/data/
- `Nf211_lattice_s.csv` — Nf=2+1+1 lattice entropy (Borsanyi et al. 2025),
  columns `T, s_over_T3, ds_over_T3`; can be regenerated from the published
  table with `export_lattice_eos.py` (needs `borsanyi2025_tab.tex`)
- `WB-chiB-1805.04445.dat` — chi_2^B and chi_4^B with errors
  (Borsanyi et al. 2018)
- `hrg_Nf211_thermo_susc.csv`, `hrg_Nf211_muhat1.csv` — Thermal-FIST HRG
  curves; model input for the HRG-constrained variant only
  (`build_controlled_hrg.py`), not for the lattice-only fit
- Legacy old-HRG model only: `Nf211_train_s.csv`, `Nf211_train_chi.csv`,
  `Nf211_nB_muhat1.csv`, `Chi2B_30-800-dT1.dat`

## Reproducing the central results

Run from the package root.

### Nf = 0
```bash
python nf0/code/train_nf0.py --n-points 16 --seeds 3
python nf0/code/plot_nf0.py
```

### Nf = 2 (seed 50, smooth-monotonic spline f)
```bash
python nf2/code/run_lqcd_training.py
```
This trains `train_nf2_smooth_monotonic_warm.py` with
`--seed 50 --pretrain-steps 20000 --phase1 6000 --phase2 3000 --w-phi 10`
and then runs the original-data, EOS and CEP evaluations.

### Nf = 2+1+1, lattice-only
```bash
python nf211/code/train_nf211_nohrg_gpu.py
python nf211/code/refine_nf211_nohrg_f_gpu.py
python nf211/code/evaluate_nf211_nohrg.py --model nf211/results/nohrg/Nf211_A_f_nohrg_best.npz --figures nf211/figures/nohrg_best
```

### Nf = 2+1+1, controlled HRG comparison (paper supplementary)
```bash
python nf211/code/build_controlled_hrg.py
python nf211/code/validate_controlled_inputs.py
python nf211/code/train_controlled_pair.py
python nf211/code/analyze_controlled_pair.py
```
The lattice-only and HRG+LQCD fits share one warm-start checkpoint,
network, grids, schedule and pressure anchor; only the low-temperature
HRG targets differ.

## Code layout notes

Each `code/` directory is self-contained:

- `emd_free.py` — EMD forward model (neural networks, horizon equations,
  thermodynamic observables); edit here to change the model structure
- `train_*.py` — training entry points. The shared training core is
  `train_nf2.py` (Nf=2) and `train_nf211.py` (Nf=2+1+1); the other
  `train_*` / `refine_*` / `scan_*` scripts are variants and refinements
  that import from them
- `build_*.py`, `extract_*.py`, `interpolate_*.py`, `export_*.py` —
  construct the training-target CSVs from raw sources
- `evaluate_*.py`, `analyze_*.py` — residuals, EOS and cusp (turning-point
  merger) evaluation of trained checkpoints
- `plot_*.py` — figure generation

Trained checkpoints are written to `results/` inside each flavor directory
(created on first run); figures go to `figures/`.
