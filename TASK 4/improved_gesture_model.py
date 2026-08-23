"""CNN hand-gesture recognizer with reproducible training and evaluation.

Expected real-data layouts are either ``TASK 4/<class>/*.jpg`` or a nested
``TASK 4/hagrid/<class>/*.jpg`` / ``TASK 4/leapGestRecog/<class>/*.jpg``.
When no real dataset is present, ``--allow-demo-data`` creates a deterministic
pattern dataset and records that fact in metrics.json; it is not presented as
real-world model performance.

Run from the repository root:
    python "TASK 4/improved_gesture_model.py" --allow-demo-data --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:  # pragma: no cover - exercised only without optional dependency
    tf = None
    keras = None


DEFAULT_CLASSES = ("fist", "palm", "peace", "ok", "one", "two", "three", "four")
IMAGE_SIZE = 96
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)


def load_folder_dataset(dataset_root: Path, max_per_class: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load images from class directories, skipping unreadable files."""
    images: list[np.ndarray] = []
    labels: list[str] = []
    class_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir()) if dataset_root.exists() else []
    if not class_dirs:
        return np.empty((0, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32), np.empty((0,), dtype=str)

    for class_dir in class_dirs:
        files = sorted(path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if max_per_class is not None:
            files = files[:max_per_class]
        for file_path in files:
            try:
                with Image.open(file_path) as image:
                    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                    images.append(np.asarray(image, dtype=np.float32) / 255.0)
                    labels.append(class_dir.name.lower())
            except (OSError, ValueError):
                continue
    return np.asarray(images, dtype=np.float32), np.asarray(labels)


def locate_real_dataset(data_dir: Path) -> Path | None:
    candidates = [data_dir, data_dir / "hagrid", data_dir / "leapGestRecog"]
    for candidate in candidates:
        if candidate.exists() and any(path.is_dir() for path in candidate.iterdir()):
            if any(path.suffix.lower() in IMAGE_EXTENSIONS for path in candidate.rglob("*")):
                return candidate
    return None


def create_demo_data(samples_per_class: int = 40, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic geometric patterns for a runnable smoke-test model."""
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, gesture in enumerate(DEFAULT_CLASSES):
        for _ in range(samples_per_class):
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (18, 18, 24))
            draw = ImageDraw.Draw(image)
            jitter_x = int(rng.integers(-4, 5))
            jitter_y = int(rng.integers(-4, 5))
            left, top, right, bottom = 18 + jitter_x, 12 + jitter_y, 78 + jitter_x, 84 + jitter_y
            draw.ellipse((left, top, right, bottom), outline=(230, 230, 230), width=3)
            if gesture == "fist":
                draw.rectangle((30 + jitter_x, 40 + jitter_y, 66 + jitter_x, 67 + jitter_y), fill=(225, 80, 80))
            elif gesture == "palm":
                draw.rectangle((37 + jitter_x, 22 + jitter_y, 59 + jitter_x, 70 + jitter_y), fill=(80, 210, 120))
                for x in (27, 36, 45, 54, 63):
                    draw.line((x + jitter_x, 15 + jitter_y, x + jitter_x, 47 + jitter_y), fill=(80, 210, 120), width=5)
            else:
                line_count = class_index - 1 if gesture in {"one", "two", "three", "four"} else class_index + 1
                if gesture == "peace":
                    line_count = 2
                if gesture == "ok":
                    draw.ellipse((29 + jitter_x, 30 + jitter_y, 67 + jitter_x, 68 + jitter_y), outline=(80, 170, 240), width=7)
                else:
                    color = (90 + (class_index * 17) % 140, 150, 240)
                    for line_index in range(max(1, min(4, line_count))):
                        x = 30 + line_index * 12 + jitter_x
                        draw.line((x, 25 + jitter_y, x, 67 + jitter_y), fill=color, width=6)
            array = np.asarray(image, dtype=np.float32) / 255.0
            noise = rng.normal(0, 0.015, array.shape).astype(np.float32)
            images.append(np.clip(array + noise, 0.0, 1.0))
            labels.append(gesture)
    return np.asarray(images, dtype=np.float32), np.asarray(labels)


def create_cnn_model(num_classes: int, input_size: int = IMAGE_SIZE):
    if keras is None:
        raise RuntimeError("TensorFlow is required for the CNN. Install dependencies with: pip install -r requirements.txt")
    model = keras.Sequential(
        [
            keras.Input(shape=(input_size, input_size, 3)),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, activation="relu"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.35),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def run(
    data_dir: Path,
    epochs: int = 10,
    batch_size: int = 32,
    max_per_class: int | None = None,
    allow_demo_data: bool = False,
    seed: int = 42,
) -> dict[str, object]:
    if keras is None:
        raise RuntimeError("TensorFlow is required for Task 4. Install dependencies with: pip install -r requirements.txt")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    seed_everything(seed)
    data_dir.mkdir(parents=True, exist_ok=True)

    real_root = locate_real_dataset(data_dir)
    if real_root is not None:
        X, labels = load_folder_dataset(real_root, max_per_class=max_per_class)
        data_source = str(real_root)
    elif allow_demo_data:
        X, labels = create_demo_data(samples_per_class=max_per_class or 40, seed=seed)
        data_source = "deterministic_demo_patterns"
    else:
        raise FileNotFoundError(
            f"No image dataset found below {data_dir}. Add class folders or rerun with --allow-demo-data."
        )

    if len(X) < 2 or len(np.unique(labels)) < 2:
        raise ValueError("At least two classes with valid images are required")
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    model = create_cnn_model(len(encoder.classes_))
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    model_path = data_dir / "gesture_cnn_model.h5"
    model.save(model_path)
    joblib.dump(encoder, data_dir / "gesture_label_encoder.joblib")
    with (data_dir / "gesture_classes.json").open("w", encoding="utf-8") as handle:
        json.dump(encoder.classes_.tolist(), handle, indent=2)

    metrics = {
        "data_source": data_source,
        "samples": int(len(X)),
        "classes": encoder.classes_.tolist(),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "epochs": epochs,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }
    with (data_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (data_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump({key: [float(value) for value in values] for key, values in history.history.items()}, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved CNN model to {model_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--allow-demo-data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_per_class=args.max_per_class,
        allow_demo_data=args.allow_demo_data,
        seed=args.seed,
    )
