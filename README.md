# BioHub Cell Tracking During Development

3D cell detection, classification, and tracking in developing zebrafish embryo microscopy —
a solution for the Kaggle
[BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
competition.

[![Tests](https://img.shields.io/badge/tests-17%20passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-CellCNN-ee4c2c)](src/model.py)
[![Kaggle](https://img.shields.io/badge/Kaggle-Biohub%20Cell%20Tracking-20beff)](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
[![Status](https://img.shields.io/badge/status-active-success)](#roadmap--future-work)
[![Version](https://img.shields.io/badge/version-v1.0.0-blue)](pyproject.toml)

**Built with:** PyTorch · NumPy · SciPy · Zarr · scikit-image · Hungarian Assignment

## Highlights

- End-to-end BioHub pipeline: dataset → detection → classification → tracking → submission
- 3D microscopy preprocessing (Zarr volumes, max-intensity projection)
- CNN candidate classification (PyTorch)
- Hungarian-assignment multi-object tracking
- Kaggle submission generation
- Modular, reusable codebase (`src/`) shared identically by notebooks and CLI
- Experiment tracking (per-run config, metrics, plots, checkpoints)
- Configuration management (one `config.py`, no hardcoded paths)
- Reproducibility (`seed_everything`, promoted checkpoint + matching normalization stats)
- Test suite (17 tests, PyTorch/tracker/dataset/seeding coverage)

<p align="center">
  <img src="docs/images/architecture.svg" alt="Pipeline architecture: Dataset to Volume Loader to Maximum Projection to Detector to CellCNN to Hungarian Tracker to submission.csv" width="440">
</p>

<p align="center">
  <img src="docs/images/pipeline_demo.gif" alt="Pipeline in action: raw frame, detected cell candidates, Hungarian tracking across frames" width="420"><br>
  <sub>Raw frame → detected candidates → Hungarian tracking across frames (real output from this repo)</sub>
</p>

## Table of Contents

- [Highlights](#highlights)
- [Competition](#competition)
- [Why This Project?](#why-this-project)
- [Project Statistics](#project-statistics)
- [Problem](#problem)
- [Architecture](#architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Roadmap](#roadmap--future-work)
- [Tech Stack](#tech-stack)
- [Citation](#citation)
- [License](#license)

## Competition

**[BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)**,
hosted by the Chan Zuckerberg Biohub. High-resolution 3D light-sheet microscopy of developing
zebrafish embryos, with the largest publicly available cell-tracking annotation set released
under CC0.

**Task**

- Detect cells in each 3D frame
- Track each cell across time
- Identify cell-division (lineage) events

**Evaluation**

The competition scores submissions on a tracking *graph* (nodes = detections, edges = links
between timepoints, forks = divisions), matched against sparse ground truth by nearest centroid
(within 7 µm):

- **Adjusted edge Jaccard** — `TP / (TP + FP + FN)` over predicted links, penalized for
  predicting far more nodes than the estimated true count
- **Division Jaccard** — same idea, scored specifically on correctly predicted division
  (parent → two daughters) events, with a ±1 timepoint tolerance

```
score = adjusted_edge_jaccard + 0.1 · division_jaccard
```

This repository currently implements detection, CNN-based candidate classification, and
frame-to-frame tracking. Cell lineage reconstruction is planned as future work — see
[Roadmap](#roadmap--future-work).

## Why This Project?

This repository demonstrates the complete workflow for a modern computer vision research
pipeline:

- 3D microscopy processing
- Candidate detection
- Deep learning classification
- Multi-object tracking
- Kaggle submission generation

This repository focuses on building a reproducible computer vision pipeline rather than
optimizing solely for leaderboard performance. Every stage lives in exactly one place in
`src/`; the notebooks are thin, readable wrappers around that same code, not a separate copy of
the logic. See [Project Statistics](#project-statistics) and [Testing](#testing) for what that
looks like in practice.

## Project Statistics

- 7 notebooks (exploration → detection → tracking → training → inference → full pipeline → tracker deep-dive)
- 13 Python modules in `src/` (config, seeding, logging, augmentations, losses, dataset, detector, tracker, metrics, experiment tracking, model, training, inference)
- 17 unit tests (`pytest`)
- PyTorch, scikit-image, scikit-learn
- Kaggle competition: BioHub - Cell Tracking During Development

## Problem

Track cells across time in 3D microscopy of developing zebrafish embryos: detect every cell in
every frame, link detections into consistent trajectories across time, and (eventually) recover
cell-division events into full lineage trees.

## Architecture

| Stage          | Method                                                     |
|----------------|--------------------------------------------------------------|
| Detection      | Gaussian smoothing + `peak_local_max` (`src/detector.py`)     |
| Classification | `CellCNN` — 3-block CNN, patch-level binary classifier (`src/model.py`) |
| Tracking       | Hungarian assignment on a pairwise distance matrix (`src/tracker.py`) |
| Submission     | Detector → CNN filter → tracker, run per-sample (`src/predict.py`, `notebooks/06_pipeline.ipynb`) |

(Diagram above, under Highlights.)

## Results

### Leaderboard

Not yet submitted to the competition — see [Roadmap](#roadmap--future-work) for what's needed
first (multi-sample/multi-frame training on the full dataset). This section will be updated
with a real score once a submission is made:

| | Score |
|---|:---:|
| Public LB | — |
| Baseline (nearest-neighbor, no CNN filter) | — |

### Demo Training Results

These metrics are from a small demonstration experiment (`experiments/demo_run/`) used only to
validate that the training loop, checkpointing, and metrics all work correctly. It trains on
patches from **one frame of one sample**, so the validation split is just 21 patches.

**They should NOT be interpreted as competition performance.**

| Metric        | Value (demo run, 21 validation patches) |
|---------------|:----------------------------------------:|
| Precision     | 1.00 |
| Recall        | 1.00 |
| F1            | 1.00 |
| ROC-AUC       | 1.00 |

<p align="center">
  <img src="docs/images/training_loss.png" alt="Training and validation loss curve" width="420">
  <img src="docs/images/training_accuracy.png" alt="Training and validation accuracy curve" width="420">
</p>
<p align="center">
  <img src="docs/images/confusion_matrix.png" alt="Validation confusion matrix" width="320">
</p>

### Detection

`CellDetector` (Gaussian smoothing + `peak_local_max`) run on a single frame's max-projection —
red markers are candidate cell centers, before CNN filtering.

<p align="center">
  <img src="docs/images/detection.png" alt="Raw detector output: candidate cell centers overlaid on a max-intensity projection" width="480">
</p>

### Tracking

`HungarianTracker` matching detections between two consecutive frames — lines connect each cell
to its optimal assignment in the next frame.

<p align="center">
  <img src="docs/images/tracking.png" alt="Hungarian assignment tracking between two frames" width="480">
</p>

## Project Structure

```
src/
    config.py            # all thresholds, paths, hyperparameters in one place
    seed.py               # seed_everything() — call once at the start of a run
    logging_utils.py       # shared logger setup
    augmentations.py        # flip / rotation / brightness / gaussian noise
    losses.py                 # BCEWithLogitsLoss + FocalLoss, selected via config
    dataset.py                 # BioHubDataset (volumes) + PatchDataset (patches)
    detector.py                  # CellDetector (Gaussian + peak_local_max)
    tracker.py                     # HungarianTracker
    metrics.py                       # tracking distance helpers + classification report
    experiment.py                      # experiments/expNNN/ tracking (config, metrics, plots, checkpoint)
    model.py                             # CellCNN
    train.py                               # training entry point: python -m src.train
    predict.py                               # inference entry point: python -m src.predict

tests/                   # pytest suite — 17 tests across model, dataset, tracker, seeding

notebooks/               # interactive walkthroughs — see below

docs/images/             # figures embedded in this README

experiments/              # one folder per training run: config, metrics, plots, checkpoint
    expNNN/

models/
    best_model.pth       # promoted checkpoint from the best training run so far
    norm_stats.npz        # matching normalization stats for that checkpoint

outputs/
    submission.csv        # generated by notebooks/05 or 06
```

## Notebooks

Each notebook is a thin, interactive layer over `src/` — the actual logic lives in one place
(`src/`) so it can run identically from a notebook or the command line.

| # | Notebook | Demonstrates |
|---|----------|---------------|
| 01 | [`01_exploration.ipynb`](notebooks/01_exploration.ipynb) | Load the dataset, inspect volume shape/dtype, visualize slices and time/z ranges |
| 02 | [`02_detection_baseline.ipynb`](notebooks/02_detection_baseline.ipynb) | Run `CellDetector` on a frame and across a full volume, detections-per-frame summary |
| 03 | [`03_tracking_baseline.ipynb`](notebooks/03_tracking_baseline.ipynb) | Naive nearest-neighbor tracking between two frames, as a baseline to compare Hungarian against |
| 04 | [`04_training_demo.ipynb`](notebooks/04_training_demo.ipynb) | Train `CellCNN` via `src/train.py`'s `run_training()`, inspect loss/accuracy curves and the confusion matrix |
| 05 | [`05_inference_submission.ipynb`](notebooks/05_inference_submission.ipynb) | Load the trained model, run detector + CNN filtering on unseen test data, build `submission.csv` |
| 06 | [`06_pipeline.ipynb`](notebooks/06_pipeline.ipynb) | Full pipeline in one notebook: detect → track (Hungarian) |
| 07 | [`07_hungarian_tracker.ipynb`](notebooks/07_hungarian_tracker.ipynb) | Hungarian assignment tracker in isolation, with a matched-pairs visualization |

## Installation

```bash
git clone https://github.com/drissou-milad/biohub-cell-tracking.git
cd biohub-cell-tracking
pip install -r requirements.txt
pip install -e .          # makes `src` importable everywhere, no sys.path hacks needed
```

Set your dataset path in `src/config.py` (`DATASET_PATH`).

## Usage

Train the CNN (no notebook required):

```bash
python -m src.train --epochs 30 --exp-name baseline
```

Each run creates `experiments/expNNN/` with its config, metrics, plots, and checkpoint, and
promotes the best checkpoint (plus its normalization stats) to `models/`. To explore
interactively instead, open the thin demo notebook, which calls the same `run_training()`
function:

```bash
jupyter notebook notebooks/04_training_demo.ipynb
```

Run inference on a sample without opening a notebook:

```bash
python -m src.predict --sample <sample_name> --split test \
    --checkpoint models/best_model.pth \
    --norm-stats models/norm_stats.npz \
    --out outputs/predictions.csv
```

(`--checkpoint` and `--norm-stats` both default to the `models/` versions above, so in most
cases you can omit them entirely.)

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

17 tests covering `CellCNN` (output shape, logits-not-probabilities, gradient flow, batch size
1), patch extraction / `PatchDataset`, `HungarianTracker` (including the classic
greedy-vs-optimal-assignment case), and `seed_everything` reproducibility.

## Roadmap / Future Work

Done:
- ✅ Dataset exploration
- ✅ Baseline detector
- ✅ Baseline tracker (Hungarian)
- ✅ CNN training pipeline (normalization, logits + `BCEWithLogitsLoss`, augmentation,
  BatchNorm/Dropout, LR scheduling, early stopping, checkpointing)
- ✅ Reusable `train.py`, experiment tracking, classification metrics
  (precision/recall/F1/ROC-AUC/confusion matrix), reproducibility (`seed_everything`), test suite
- ✅ End-to-end inference notebook (detector → CNN filter → submission.csv)

Not yet implemented:
- ⬜ Train on multiple samples / multiple frames, not just frame 0 of one sample
- ⬜ Evaluate against the full competition dataset and record an actual leaderboard score
- ⬜ Richer tracking cost matrix (distance + size + intensity + motion prediction)
- ⬜ Kalman filter for motion prediction between frames
- ⬜ Cell division / lineage detection (parent → child1, child2) — required for the competition's division Jaccard term
- ⬜ Better detection (adaptive threshold, distance transform, watershed)
- ⬜ Cross-validation, test-time augmentation, model ensembling
- ⬜ Transfer learning (ResNet/EfficientNet backbone) as an alternative to `CellCNN`
- ⬜ Streamlit dashboard for interactive TIFF upload → detect → track → export

## Tech Stack

Python, NumPy, SciPy, Zarr, PyTorch, scikit-image, scikit-learn, Matplotlib, Pandas

## Citation

No accompanying report or paper yet — this section will be filled in if one is written.

## License

[MIT](LICENSE)
