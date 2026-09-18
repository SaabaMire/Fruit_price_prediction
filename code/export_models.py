"""Export the trained scikit-learn models for the static browser app."""

import json
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
OUTPUT = ROOT / "docs" / "model.json"


def export_tree(tree):
    return {
        "childrenLeft": tree.children_left.tolist(),
        "childrenRight": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": tree.value[:, 0, 0].tolist(),
    }


def main():
    linear = joblib.load(MODEL_DIR / "lr_fruit_model.pkl")
    forest = joblib.load(MODEL_DIR / "rf_fruit_model.pkl")

    feature_names = linear.feature_names_in_.tolist()
    if feature_names != forest.feature_names_in_.tolist():
        raise ValueError("The trained models use different feature orders.")

    payload = {
        "featureNames": feature_names,
        "linear": {
            "intercept": float(linear.intercept_),
            "coefficients": linear.coef_.tolist(),
        },
        "forest": {
            "trees": [export_tree(estimator.tree_) for estimator in forest.estimators_]
        },
    }

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {len(forest.estimators_)} trees to {OUTPUT}")


if __name__ == "__main__":
    main()
