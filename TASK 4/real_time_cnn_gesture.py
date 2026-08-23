"""Live webcam inference for the trained Task 4 gesture CNN.

Run from any working directory after training:
    python "TASK 4/real_time_cnn_gesture.py"
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tensorflow as tf


DEFAULT_IMAGE_SIZE = 96


def load_artifacts(model_dir: Path):
    model_path = model_dir / "gesture_cnn_model.h5"
    encoder_path = model_dir / "gesture_label_encoder.joblib"
    classes_path = model_dir / "gesture_classes.json"
    missing = [str(path) for path in (model_path, encoder_path, classes_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: " + ", ".join(missing) + ". Run improved_gesture_model.py first."
        )
    model = tf.keras.models.load_model(model_path)
    encoder = joblib.load(encoder_path)
    class_names = json.loads(classes_path.read_text(encoding="utf-8"))
    return model, encoder, class_names


def preprocess_hand_region(image: np.ndarray, landmarks, image_size: int):
    height, width, _ = image.shape
    x_coords = [landmark.x for landmark in landmarks.landmark]
    y_coords = [landmark.y for landmark in landmarks.landmark]
    padding = 30
    x_min = max(0, int(min(x_coords) * width) - padding)
    y_min = max(0, int(min(y_coords) * height) - padding)
    x_max = min(width, int(max(x_coords) * width) + padding)
    y_max = min(height, int(max(y_coords) * height) + padding)
    hand_region = image[y_min:y_max, x_min:x_max]
    if hand_region.size == 0:
        return None
    hand_region = cv2.resize(hand_region, (image_size, image_size)).astype(np.float32) / 255.0
    return hand_region, (x_min, y_min, x_max, y_max)


def run(model_dir: Path, camera_index: int = 0, image_size: int = DEFAULT_IMAGE_SIZE) -> None:
    model, encoder, class_names = load_artifacts(model_dir)
    hands_module = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    history: deque[str] = deque(maxlen=5)
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    try:
        with hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                success, frame = camera.read()
                if not success:
                    break
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                gesture_name, confidence, bbox = "No Hand", 0.0, None
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    drawing.draw_landmarks(frame, hand_landmarks, hands_module.HAND_CONNECTIONS)
                    processed = preprocess_hand_region(rgb_frame, hand_landmarks, image_size)
                    if processed is not None:
                        hand_region, bbox = processed
                        predictions = model.predict(np.expand_dims(hand_region, axis=0), verbose=0)[0]
                        class_index = int(np.argmax(predictions))
                        gesture_name = str(encoder.inverse_transform([class_index])[0])
                        confidence = float(predictions[class_index])
                        history.append(gesture_name)
                        gesture_name = Counter(history).most_common(1)[0][0]

                if bbox is not None:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                label = f"{gesture_name.upper()} ({confidence:.2f})" if confidence > 0.5 else "UNCERTAIN"
                cv2.putText(frame, label, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"CNN | {len(class_names)} classes | Press q to quit",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow("CNN Hand Gesture Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.model_dir, args.camera_index, args.image_size)
