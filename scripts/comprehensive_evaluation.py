"""
Comprehensive classifier evaluation with statistical analysis and
publication-quality figures.

Trains all 6 traditional ML classifiers on extracted plant-disease features,
computes a full suite of metrics (accuracy, precision, recall, F1, ROC-AUC),
runs stratified k-fold cross-validation with 95 % confidence intervals,
analyses feature importance via multiple methods, and produces carefully
designed figures and tables.

Output artefacts (saved to ``exports/evaluation/``):
  - metrics_summary.csv          — per-model metric table
  - per_class_metrics.csv        — per-class precision / recall / F1
  - cross_validation_scores.csv  — per-fold accuracy for each model
  - feature_importance.csv       — aggregated feature importance ranking
  - fig_metric_comparison.png    — grouped bar chart of all metrics
  - fig_confusion_matrices.png   — one heatmap per model (single figure)
  - fig_roc_curves.png           — per-class & macro-average ROC per model
  - fig_feature_importance.png   — top-20 features (RF, XGBoost, permutation)

Usage
-----
    python scripts/comprehensive_evaluation.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelBinarizer, label_binarize

# ---------------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib
from catboost import CatBoostClassifier

from plantdisease.models.utils import load_features
from plantdisease.models.train_rf import RFEnsembleClassifier
from plantdisease.models import train_svm, train_logistic_regression, train_knn
from plantdisease.models import train_xgboost, train_catboost

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURES_PATH = "data/processed/features.csv"
OUTPUT_DIR = Path("exports/evaluation")
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_CV_FOLDS = 5
TOP_N_FEATURES = 20
FIG_DPI = 300

# Faster CatBoost settings (200 iters + higher LR converges similarly to 800 + 0.03)
CATBOOST_ITERATIONS = 200
CATBOOST_LR = 0.1

# Colour palette – one colour per model, colour-blind safe
MODEL_COLOURS = {
    "Random Forest": "#4C72B0",
    "XGBoost": "#DD8452",
    "CatBoost": "#55A868",
    "SVM (RBF)": "#C44E52",
    "Logistic Regression": "#8172B3",
    "KNN": "#937860",
}


# ============================================================================
# 1.  DATA LOADING
# ============================================================================

def load_data():
    """Return X, y (encoded ints), class_names, and feature_names."""
    X, y, class_names = load_features(FEATURES_PATH)
    df = pd.read_csv(FEATURES_PATH)
    feature_names = df.drop(columns=["label", "image_id"], errors="ignore").columns.values
    return X, y, class_names, feature_names


# ============================================================================
# 2.  CLASSIFIER TRAINING
# ============================================================================

def _train_catboost_fast(X_train, y_train, iterations=CATBOOST_ITERATIONS, lr=CATBOOST_LR):
    """Train CatBoost with fewer iterations + higher LR for speed."""
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=lr,
        depth=10,
        l2_leaf_reg=1.0,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=RANDOM_STATE,
        task_type="CPU",
        verbose=0,
    )
    model.fit(X_train, y_train)
    return model


def _train_all(X_train, y_train):
    """Train every classifier and return {name: model} dict."""
    models = {}

    print("  Random Forest...")
    rf = RFEnsembleClassifier(n_estimators=500)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    print("  XGBoost...")
    models["XGBoost"] = train_xgboost.train(X_train, y_train)

    print("  CatBoost...")
    models["CatBoost"] = _train_catboost_fast(X_train, y_train)

    print("  SVM...")
    models["SVM (RBF)"] = train_svm.train(X_train, y_train)

    print("  Logistic Regression...")
    models["Logistic Regression"] = train_logistic_regression.train(X_train, y_train)

    print("  KNN...")
    models["KNN"] = train_knn.train(X_train, y_train)

    return models


# ============================================================================
# 3.  METRIC COMPUTATION
# ============================================================================

def _compute_metrics(model, X_test, y_test, class_names):
    """Compute a full metric dict for a single trained model."""
    y_pred = model.predict(X_test)

    # Probabilities — needed for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    # ROC-AUC (one-vs-rest, macro & weighted)
    if y_prob is not None:
        n_classes = len(class_names)
        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
        if y_prob.shape[1] == n_classes:
            metrics["roc_auc_macro"] = roc_auc_score(
                y_test_bin, y_prob, average="macro", multi_class="ovr",
            )
            metrics["roc_auc_weighted"] = roc_auc_score(
                y_test_bin, y_prob, average="weighted", multi_class="ovr",
            )
        else:
            metrics["roc_auc_macro"] = np.nan
            metrics["roc_auc_weighted"] = np.nan
    else:
        metrics["roc_auc_macro"] = np.nan
        metrics["roc_auc_weighted"] = np.nan

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred)

    # Per-class report
    report = classification_report(y_test, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    metrics["per_class"] = report

    # Store predictions / probabilities for later plots
    metrics["y_pred"] = y_pred
    metrics["y_prob"] = y_prob

    return metrics


# ============================================================================
# 4.  CROSS-VALIDATION WITH CONFIDENCE INTERVALS
# ============================================================================

def _fit_rf(X_tr, y_tr):
    """Helper: create and fit an RF for CV folds."""
    rf = RFEnsembleClassifier(n_estimators=500)
    rf.fit(X_tr, y_tr)
    return rf


def _cross_validate_all(X, y, n_splits=N_CV_FOLDS):
    """Run stratified k-fold CV for every classifier and return a DataFrame."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    classifiers = {
        "Random Forest": lambda Xtr, ytr: _fit_rf(Xtr, ytr),
        "XGBoost": train_xgboost.train,
        "CatBoost": lambda Xtr, ytr: _train_catboost_fast(Xtr, ytr),
        "SVM (RBF)": train_svm.train,
        "Logistic Regression": train_logistic_regression.train,
        "KNN": train_knn.train,
    }

    rows = []
    for name, factory in classifiers.items():
        fold_scores = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model = factory(X_tr, y_tr)

            y_pred = model.predict(X_te)
            acc = accuracy_score(y_te, y_pred)
            fold_scores.append(acc)
            rows.append({"model": name, "fold": fold_idx + 1, "accuracy": acc})

        arr = np.array(fold_scores)
        mean, std = arr.mean(), arr.std()
        n = len(arr)
        ci95 = stats.t.interval(0.95, df=n - 1, loc=mean, scale=std / np.sqrt(n))
        print(f"  {name:<25}  mean={mean:.4f}  std={std:.4f}  "
              f"95% CI=[{ci95[0]:.4f}, {ci95[1]:.4f}]")

    return pd.DataFrame(rows)


# ============================================================================
# 5.  FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def _feature_importance(models, X_test, y_test, feature_names):
    """Aggregate feature importance from multiple methods.

    Sources:
        - Random Forest Gini importance
        - XGBoost gain importance
        - CatBoost importance
        - Permutation importance (averaged across all models)
    """
    importance_df = pd.DataFrame({"feature": feature_names})

    # --- Tree-based native importance ---
    # RF (uses internal scaler, so importance is on scaled features — still valid for ranking)
    rf_model = models["Random Forest"]
    importance_df["rf_importance"] = rf_model.rf.feature_importances_

    # XGBoost
    xgb_model = models["XGBoost"]
    importance_df["xgb_importance"] = xgb_model.feature_importances_

    # CatBoost
    cb_model = models["CatBoost"]
    cb_imp = cb_model.get_feature_importance()
    importance_df["catboost_importance"] = cb_imp / cb_imp.sum()  # normalise

    # --- Permutation importance (model-agnostic) ---
    print("\n  Computing permutation importance (this may take a moment)...")
    perm_scores = np.zeros(len(feature_names))
    perm_count = 0
    for name, model in models.items():
        try:
            result = permutation_importance(
                model, X_test, y_test,
                n_repeats=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                scoring="accuracy",
            )
            perm_scores += result.importances_mean
            perm_count += 1
        except Exception:
            pass

    if perm_count > 0:
        importance_df["permutation_importance"] = perm_scores / perm_count
    else:
        importance_df["permutation_importance"] = 0.0

    # --- Aggregate rank ---
    rank_cols = ["rf_importance", "xgb_importance", "catboost_importance",
                 "permutation_importance"]
    for col in rank_cols:
        importance_df[f"{col}_rank"] = importance_df[col].rank(ascending=False)

    importance_df["mean_rank"] = importance_df[[f"{c}_rank" for c in rank_cols]].mean(axis=1)
    importance_df = importance_df.sort_values("mean_rank")

    return importance_df


# ============================================================================
# 6.  FIGURES
# ============================================================================

def _plot_metric_comparison(summary_df, out_path):
    """Grouped bar chart comparing macro metrics across classifiers."""
    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"]
    display_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    plot_df = summary_df.set_index("model")[metric_cols].copy()
    plot_df.columns = display_names

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot.bar(ax=ax, width=0.75, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Score")
    ax.set_title("Classifier Performance Comparison (Macro-Averaged)")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="lower right", frameon=True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%",
                     label_type="edge", fontsize=6, padding=2,
                     labels=[f"{v * 100:.1f}" for v in container.datavalues])

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_confusion_matrices(all_metrics, class_names, out_path):
    """One confusion-matrix heatmap per model in a single figure."""
    n = len(all_metrics)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_2d(axes)

    short_names = [c.replace("___", "\n").replace(",_", "\n") for c in class_names]

    for idx, (name, m) in enumerate(all_metrics.items()):
        ax = axes[idx // cols, idx % cols]
        cm = m["confusion_matrix"]

        # Normalise to percentages per true class
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=short_names, yticklabels=short_names,
                    ax=ax, cbar=False, vmin=0, vmax=100,
                    annot_kws={"size": 8})
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(labelsize=7)

    # Remove unused subplots
    for idx in range(len(all_metrics), rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("Normalised Confusion Matrices (%)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_roc_curves(all_metrics, class_names, out_path):
    """Per-model ROC curves (macro-average + per-class) in a single figure."""
    n_classes = len(class_names)
    models_with_prob = {k: v for k, v in all_metrics.items() if v["y_prob"] is not None}

    if not models_with_prob:
        print("  Skipping ROC curves — no models provide probability estimates.")
        return

    n = len(models_with_prob)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows))
    axes = np.atleast_2d(axes)

    for idx, (name, m) in enumerate(models_with_prob.items()):
        ax = axes[idx // cols, idx % cols]
        y_test = m["_y_test"]
        y_prob = m["y_prob"]

        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

        # Per-class ROC
        fpr_all, tpr_all, roc_auc_all = {}, {}, {}
        for i in range(n_classes):
            fpr_all[i], tpr_all[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc_all[i] = auc(fpr_all[i], tpr_all[i])

        # Macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr_all[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_all[i], tpr_all[i])
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)

        # Plot per-class with transparency
        for i in range(n_classes):
            ax.plot(fpr_all[i], tpr_all[i], alpha=0.25, linewidth=0.8)

        ax.plot(all_fpr, mean_tpr, color="navy", linewidth=2,
                label=f"Macro-avg (AUC = {macro_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)

    for idx in range(len(models_with_prob), rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("ROC Curves (One-vs-Rest)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_feature_importance(importance_df, out_path, top_n=TOP_N_FEATURES):
    """Horizontal bar chart of top features from multiple methods."""
    top = importance_df.head(top_n).copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    method_cols = [
        ("rf_importance", "Random Forest (Gini)", "#4C72B0"),
        ("xgb_importance", "XGBoost (Gain)", "#DD8452"),
        ("permutation_importance", "Permutation (Avg)", "#55A868"),
    ]

    for ax, (col, title, colour) in zip(axes, method_cols):
        sorted_top = top.sort_values(col, ascending=True)
        ax.barh(sorted_top["feature"], sorted_top[col], color=colour, edgecolor="white")
        ax.set_xlabel("Importance")
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)

    fig.suptitle(f"Top {top_n} Features by Importance", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ============================================================================
# 7.  TABLE EXPORTS
# ============================================================================

def _save_tables(summary_df, per_class_rows, cv_df, importance_df, out_dir):
    """Save metric tables to CSV."""
    summary_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    print(f"  Saved {out_dir / 'metrics_summary.csv'}")

    pd.DataFrame(per_class_rows).to_csv(out_dir / "per_class_metrics.csv", index=False)
    print(f"  Saved {out_dir / 'per_class_metrics.csv'}")

    cv_df.to_csv(out_dir / "cross_validation_scores.csv", index=False)
    print(f"  Saved {out_dir / 'cross_validation_scores.csv'}")

    importance_df.to_csv(out_dir / "feature_importance.csv", index=False)
    print(f"  Saved {out_dir / 'feature_importance.csv'}")


# ============================================================================
# 8.  CONSOLE SUMMARY
# ============================================================================

def _print_summary(summary_df, cv_df):
    """Print a summary table to stdout."""
    print("\n" + "=" * 90)
    print("EVALUATION SUMMARY  (test-set metrics, macro-averaged)")
    print("=" * 90)

    display_cols = ["model", "accuracy", "precision_macro", "recall_macro",
                    "f1_macro", "roc_auc_macro"]
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    fmt_df = summary_df[display_cols].copy()
    for c in display_cols[1:]:
        fmt_df[c] = fmt_df[c].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    fmt_df.columns = headers
    print(fmt_df.to_string(index=False))

    # CV summary
    print("\n" + "-" * 90)
    print(f"STRATIFIED {N_CV_FOLDS}-FOLD CROSS-VALIDATION  (accuracy)")
    print("-" * 90)
    for name, grp in cv_df.groupby("model"):
        arr = grp["accuracy"].values
        mean, std = arr.mean(), arr.std()
        n = len(arr)
        ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=std / np.sqrt(n))
        print(f"  {name:<25}  {mean:.4f} +/- {std:.4f}   95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    print("=" * 90)


# ============================================================================
# MAIN
# ============================================================================

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("[1/6] Loading features...")
    X, y, class_names, feature_names = load_data()
    print(f"      {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )

    # ---- Train models ----
    print("[2/6] Training all classifiers...")
    models = _train_all(X_train, y_train)

    # ---- Evaluate ----
    print("\n[3/6] Computing metrics on held-out test set...")
    all_metrics = {}
    summary_rows = []
    per_class_rows = []

    for name, model in models.items():
        m = _compute_metrics(model, X_test, y_test, class_names)
        m["_y_test"] = y_test   # needed for ROC plots
        all_metrics[name] = m

        row = {"model": name}
        for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                     "precision_weighted", "recall_weighted", "f1_weighted",
                     "roc_auc_macro", "roc_auc_weighted"]:
            row[key] = m[key]
        summary_rows.append(row)

        # Per-class rows
        for cls_name in class_names:
            cls_key = str(cls_name)
            if cls_key in m["per_class"]:
                per_class_rows.append({
                    "model": name,
                    "class": cls_name,
                    "precision": m["per_class"][cls_key]["precision"],
                    "recall": m["per_class"][cls_key]["recall"],
                    "f1": m["per_class"][cls_key]["f1-score"],
                    "support": m["per_class"][cls_key]["support"],
                })

    summary_df = pd.DataFrame(summary_rows)

    # ---- Cross-validation ----
    print(f"\n[4/6] Stratified {N_CV_FOLDS}-fold cross-validation...")
    cv_df = _cross_validate_all(X, y, n_splits=N_CV_FOLDS)

    # ---- Feature importance ----
    print("\n[5/6] Analysing feature importance...")
    importance_df = _feature_importance(models, X_test, y_test, feature_names)

    # ---- Figures & tables ----
    print("\n[6/6] Generating figures and tables...")

    _plot_metric_comparison(summary_df, OUTPUT_DIR / "fig_metric_comparison.png")
    _plot_confusion_matrices(all_metrics, class_names, OUTPUT_DIR / "fig_confusion_matrices.png")
    _plot_roc_curves(all_metrics, class_names, OUTPUT_DIR / "fig_roc_curves.png")
    _plot_feature_importance(importance_df, OUTPUT_DIR / "fig_feature_importance.png")
    _save_tables(summary_df, per_class_rows, cv_df, importance_df, OUTPUT_DIR)

    # ---- Console summary ----
    _print_summary(summary_df, cv_df)

    print(f"\nAll artefacts saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
