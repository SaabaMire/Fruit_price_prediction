"""Verify that browser model data matches scikit-learn predictions."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def predict_tree(tree, features):
    node = 0
    while tree["childrenLeft"][node] != -1:
        feature = tree["feature"][node]
        # JavaScript's Math.fround returns a double containing the float32 value.
        browser_value = float(np.float32(features[feature]))
        if browser_value <= tree["threshold"][node]:
            node = tree["childrenLeft"][node]
        else:
            node = tree["childrenRight"][node]
    return tree["value"][node]


def main():
    payload = json.loads((ROOT / "docs" / "model.json").read_text(encoding="utf-8"))
    linear = joblib.load(ROOT / "model" / "lr_fruit_model.pkl")
    forest = joblib.load(ROOT / "model" / "rf_fruit_model.pkl")
    features = pd.read_csv(ROOT / "dataset" / "Clean_fruit_dataset_encoded.csv").drop(columns=["price"]).head(50)
    values = features.astype(float).to_numpy()

    browser_linear = payload["linear"]["intercept"] + values @ np.array(payload["linear"]["coefficients"])
    browser_forest = np.array([
        np.mean([predict_tree(tree, row) for tree in payload["forest"]["trees"]])
        for row in values
    ])

    np.testing.assert_allclose(browser_linear, linear.predict(features), atol=1e-12)
    np.testing.assert_allclose(browser_forest, forest.predict(features), atol=1e-12)
    print("Browser exports match both Python models for 50 sample rows.")


if __name__ == "__main__":
    main()
