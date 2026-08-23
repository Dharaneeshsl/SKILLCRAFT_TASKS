"""Linear regression baseline for the Ames house-price dataset.

Run from any working directory:
    python "TASK 1/linear_regression_model.py"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath"]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the four requested predictors plus a combined bathroom feature."""
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    result = frame[FEATURES].copy()
    result["TotalBath"] = result["FullBath"] + 0.5 * result["HalfBath"]
    return result.drop(columns=["FullBath", "HalfBath"])


def run(data_dir: Path, test_size: float = 0.2, random_state: int = 42) -> dict[str, float]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X = build_features(train_df)
    y = train_df["SalePrice"]
    X_test = build_features(test_df)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    validation_model = LinearRegression().fit(X_train, y_train)
    validation_predictions = validation_model.predict(X_valid)
    metrics = {
        "validation_rmse": float(mean_squared_error(y_valid, validation_predictions) ** 0.5),
        "validation_r2": float(r2_score(y_valid, validation_predictions)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": FEATURES + ["TotalBath"],
    }

    model = LinearRegression().fit(X, y)
    predictions = model.predict(X_test)
    pd.DataFrame({"Id": test_df["Id"], "SalePrice": predictions}).to_csv(
        data_dir / "submission.csv", index=False
    )

    fitted_predictions = model.predict(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(y, fitted_predictions, alpha=0.55, edgecolors="none")
    limits = [float(y.min()), float(y.max())]
    plt.plot(limits, limits, "r--", linewidth=1.5, label="Ideal prediction")
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("Actual vs. Predicted SalePrice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(data_dir / "actual_vs_predicted.png", dpi=160)
    plt.close()

    with (data_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved {len(predictions)} predictions to {data_dir / 'submission.csv'}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing train.csv and test.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir)
