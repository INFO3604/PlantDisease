"""
Evaluate all 6 traditional classifiers on the extracted features.
Prints a summary table with accuracy for each classifier.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plantdisease.models.utils import load_features
from plantdisease.models.train_rf import RFEnsembleClassifier
from plantdisease.models import train_svm, train_logistic_regression, train_knn
from plantdisease.models import train_xgboost, train_catboost

FEATURES_PATH = "data/processed/features.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    X, y, class_names = load_features(FEATURES_PATH)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )

    results = {}

    # 1. Random Forest
    print("Training Random Forest...")
    rf = RFEnsembleClassifier()
    rf.fit(X_train, y_train)
    rf_acc = np.mean(rf.predict(X_test) == y_test)
    results["Random Forest"] = rf_acc
    print(f"  Accuracy: {rf_acc:.4f}")

    # 2. SVM (RBF)
    print("Training SVM (RBF)...")
    svm_model = train_svm.train(X_train, y_train)
    svm_preds = train_svm.predict(svm_model, X_test)
    svm_acc = np.mean(svm_preds == y_test)
    results["SVM (RBF)"] = svm_acc
    print(f"  Accuracy: {svm_acc:.4f}")

    # 3. Logistic Regression
    print("Training Logistic Regression...")
    lr_model = train_logistic_regression.train(X_train, y_train)
    lr_preds = train_logistic_regression.predict(lr_model, X_test)
    lr_acc = np.mean(lr_preds == y_test)
    results["Logistic Regression"] = lr_acc
    print(f"  Accuracy: {lr_acc:.4f}")

    # 4. KNN
    print("Training KNN...")
    knn_model = train_knn.train(X_train, y_train)
    knn_preds = train_knn.predict(knn_model, X_test)
    knn_acc = np.mean(knn_preds == y_test)
    results["KNN"] = knn_acc
    print(f"  Accuracy: {knn_acc:.4f}")

    # 5. XGBoost
    print("Training XGBoost...")
    xgb_model = train_xgboost.train(X_train, y_train)
    xgb_preds = train_xgboost.predict(xgb_model, X_test)
    xgb_acc = np.mean(xgb_preds == y_test)
    results["XGBoost"] = xgb_acc
    print(f"  Accuracy: {xgb_acc:.4f}")

    # 6. CatBoost
    print("Training CatBoost...")
    cb_model = train_catboost.train(X_train, y_train)
    cb_preds = train_catboost.predict(cb_model, X_test)
    cb_acc = np.mean(cb_preds == y_test)
    results["CatBoost"] = cb_acc
    print(f"  Accuracy: {cb_acc:.4f}")

    # Summary
    print("\n" + "=" * 50)
    print("CLASSIFIER ACCURACY SUMMARY")
    print("=" * 50)
    print(f"{'Classifier':<25} {'Accuracy':>10}")
    print("-" * 37)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ✓" if acc >= 0.85 else " ✗"
        print(f"{name:<25} {acc:>9.2%}{marker}")
    print("-" * 37)
    all_above = all(a >= 0.85 for a in results.values())
    print(f"All ≥ 85%: {'YES' if all_above else 'NO'}")


if __name__ == "__main__":
    main()
