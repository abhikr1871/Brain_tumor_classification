"""
Classical Machine Learning Baselines for Brain Tumor Classification.
Uses extracted handcrafted features (GLCM Haralick + LBP + Intensity Moments).
Algorithms:
- Support Vector Machine (SVM with RBF Kernel)
- Random Forest Classifier
"""

from typing import List, Optional
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

def build_svm_pipeline(C: float = 10.0, gamma: str = "scale") -> Pipeline:
    """Creates a standard scaled SVM classification pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=C, kernel="rbf", gamma=gamma, probability=True, random_state=42))
    ])

def build_rf_pipeline(n_estimators: int = 150) -> Pipeline:
    """Creates a Random Forest classification pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=n_estimators, max_depth=12, random_state=42))
    ])

def train_and_eval_classical(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
):
    """Trains and compares both SVM and Random Forest baselines."""
    results = {}

    models = {
        "SVM (RBF)": build_svm_pipeline(),
        "Random Forest": build_rf_pipeline(),
    }

    for name, clf in models.items():
        print(f"[*] Training {name} on handcrafted features...")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rep_text = classification_report(y_test, preds, target_names=class_names, output_dict=False)
        rep_dict = classification_report(y_test, preds, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_test, preds)
        print(f"    --> {name} Test Accuracy: {acc*100:.2f}%")
        print(rep_text)
        results[name] = {
            "accuracy": acc,
            "report": rep_dict,
            "report_text": rep_text,
            "confusion_matrix": cm,
            "predictions": preds,
            "model": clf,
        }

    return results
