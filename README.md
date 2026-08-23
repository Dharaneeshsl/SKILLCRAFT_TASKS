# SkillCraft Machine-Learning Tasks

This repository contains four reproducible machine-learning assignments. Each task has a self-contained script, uses paths relative to its own location, saves non-interactive visualizations, and writes machine-readable metrics.

## Task summary

| Task | Method | Dataset | Main outputs |
|---|---|---|---|
| 1 | Linear regression | Ames house prices | `submission.csv`, `metrics.json`, `actual_vs_predicted.png` |
| 2 | K-Means clustering | Mall Customers | `customer_segments.csv`, `metrics.json`, `elbow_method.png`, `kmeans_clusters.png` |
| 3 | Linear SVM image classification | Microsoft Cats vs. Dogs | `svm_cats_vs_dogs.joblib`, `metrics.json`, `confusion_matrix.png`, `predictions_grid.png` |
| 4 | CNN image classification with live webcam inference | User-provided gesture folders | `gesture_cnn_model.h5`, `gesture_classes.json`, `metrics.json`, `training_history.json` |

## Setup

Use Python 3.10 or newer, create a virtual environment, and install the dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the tasks

```bash
python "TASK 1/linear_regression_model.py"
python "TASK 2/kmeans_clustering.py"
python "TASK 3/svm_cats_vs_dogs.py"
```

Task 3 uses 1,000 valid images per class by default. This is deliberate so the repository remains practical to run while retaining the supplied dataset. Use `--samples-per-class` to change the limit.

Task 4 looks for class folders directly below `TASK 4`, `TASK 4/hagrid`, or `TASK 4/leapGestRecog`. For a real training run, place images in folders such as `TASK 4/palm/*.jpg` and `TASK 4/fist/*.jpg`, then run:

```bash
python "TASK 4/improved_gesture_model.py" --epochs 10
python "TASK 4/real_time_cnn_gesture.py"
```

A real gesture dataset is not included in the repository. For a reproducible smoke test only, the training script supports an explicit deterministic demo dataset:

```bash
python "TASK 4/improved_gesture_model.py" --allow-demo-data --epochs 3 --max-per-class 40
```

The resulting `TASK 4/metrics.json` records `data_source: deterministic_demo_patterns`, so demo accuracy is not confused with real-world performance.

## Reproducibility and outputs

All scripts use fixed random seeds by default. Generated metrics and plots are written beside the script that produced them. Training artifacts are kept inside `TASK 4` so the webcam demo can load them without relying on the current working directory.

The Microsoft Cats vs. Dogs dataset is retained for the educational, non-commercial classification task under the accompanying license agreement. Do not redistribute the raw image dataset outside the terms that apply to it.
