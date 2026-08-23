"""SVM image classifier for the Microsoft Cats vs. Dogs dataset.

Run from any working directory:
    python "TASK 3/svm_cats_vs_dogs.py"

The model uses normalized HOG descriptors and a nonlinear SVM. It scans all
files until the requested valid-image limit is met and skips corrupt files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


IMAGE_SIZE = (64, 64)
CLASS_NAMES = ("Cat", "Dog")


def load_images(
    folder: Path, label: int, max_samples: int
) -> tuple[list[np.ndarray], list[int], list[np.ndarray], int]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    previews: list[np.ndarray] = []
    invalid = 0
    for file_path in sorted(folder.iterdir()):
        if len(features) >= max_samples:
            break
        if not file_path.is_file() or file_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            with Image.open(file_path) as image:
                image = image.convert("L").resize(IMAGE_SIZE)
                pixels = np.asarray(image, dtype=np.float32) / 255.0
                features.append(
                    hog(
                        pixels,
                        orientations=9,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        block_norm="L2-Hys",
                    ).astype(np.float32)
                )
                previews.append(pixels)
                labels.append(label)
        except (OSError, ValueError):
            invalid += 1
    return features, labels, previews, invalid


def run(
    data_dir: Path,
    output_dir: Path | None = None,
    samples_per_class: int = 1000,
    random_state: int = 42,
) -> dict[str, object]:
    output_dir = output_dir or data_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2")
    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    all_previews: list[np.ndarray] = []
    invalid_images = 0
    for label, class_name in enumerate(CLASS_NAMES):
        features, labels, previews, invalid = load_images(
            data_dir / class_name, label, samples_per_class
        )
        if len(features) < 2:
            raise ValueError(f"Need at least two valid images in {class_name}; found {len(features)}")
        all_features.extend(features)
        all_labels.extend(labels)
        all_previews.extend(previews)
        invalid_images += invalid

    X = np.asarray(all_features, dtype=np.float32)
    y = np.asarray(all_labels, dtype=np.int64)
    previews = np.asarray(all_previews, dtype=np.float32)
    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=random_state, stratify=y
    )
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=10.0, kernel="rbf", gamma="scale", random_state=random_state)),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    metrics = {
        "feature_extractor": "HOG",
        "samples_per_class": samples_per_class,
        "valid_samples": int(len(X)),
        "invalid_images_skipped": invalid_images,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }

    joblib.dump(model, output_dir / "svm_cats_vs_dogs.joblib")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(f"Confusion Matrix (accuracy={accuracy:.3f})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    threshold = cm.max() / 2 if cm.size else 0
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                int(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    display_count = min(16, len(test_indices))
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index >= display_count:
            continue
        test_position = index
        ax.imshow(previews[test_indices[test_position]], cmap="gray")
        predicted = CLASS_NAMES[int(y_pred[test_position])]
        actual = CLASS_NAMES[int(y_test[test_position])]
        ax.set_title(f"P: {predicted} | T: {actual}", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "predictions_grid.png", dpi=160)
    plt.close(fig)

    print(json.dumps({"accuracy": accuracy, "valid_samples": len(X), "invalid_images_skipped": invalid_images}, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "PetImages")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--samples-per-class", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, output_dir=args.output_dir, samples_per_class=args.samples_per_class)
