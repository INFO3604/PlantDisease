"""Backward-compatibility shim — re-exports from train_rf."""

from .train_rf import RFEnsembleClassifier  # noqa: F401


def train_rf_ensemble(
    train_features_path,
    val_features_path=None,
    output_dir=None,
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
):
    """Train a Random Forest ensemble and return (classifier, metrics).

    Convenience wrapper used by ``scripts/train_rf_cli.py``.
    """
    import numpy as np
    from pathlib import Path

    data = np.load(train_features_path, allow_pickle=True)
    X_train = data["features"].astype(np.float32)
    y_train = data["labels"]
    feature_names = (
        data["feature_names"].tolist() if "feature_names" in data else None
    )

    X_val, y_val = None, None
    if val_features_path is not None:
        vdata = np.load(val_features_path, allow_pickle=True)
        X_val = vdata["features"].astype(np.float32)
        y_val = vdata["labels"]

    clf = RFEnsembleClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val, feature_names=feature_names)

    metrics = {}
    if X_val is not None and y_val is not None:
        metrics = clf.evaluate(X_val, y_val)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        clf.save(out / "rf_model")

    return clf, metrics
