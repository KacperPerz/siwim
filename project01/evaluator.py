import numpy as np
import matplotlib.pyplot as plt

from typing import Dict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
)
from config import ModelConfig, TestingConfig, RANDOM_STATE
from super_classifier import SuperClassifier


class ClassifierEvaluator:
    def __init__(
        self,
        baseline_config: ModelConfig,
        target_config: ModelConfig,
        random_state=RANDOM_STATE,
    ):
        self.random_state = random_state
        self.results = {
            "baseline": {
                "auprc": [],
                "auroc": [],
                "fnr_medium": [],
                "fpr_medium": [],
                "fnr_high": [],
                "fpr_high": [],
                "neg_percent": [],
                "pos_percent": [],
            },
            "target": {
                "auprc": [],
                "auroc": [],
                "fnr_medium": [],
                "fpr_medium": [],
                "fnr_high": [],
                "fpr_high": [],
                "neg_percent": [],
                "pos_percent": [],
            },
        }
        self.baseline_config = baseline_config
        self.target_config = target_config

    def _find_threshold(
        self, y_true, y_scores, criterion="sensitivity", target_value=0.99
    ):
        """Find threshold based on sensitivity or specificity target."""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        if criterion == "sensitivity":
            valid_indices = np.where(tpr >= target_value)[0]
            if len(valid_indices) > 0:
                idx = valid_indices[0]
                return thresholds[idx]
            else:
                return min(thresholds)
        elif criterion == "specificity":
            specificity = 1 - fpr
            valid_indices = np.where(specificity >= target_value)[0]
            if len(valid_indices) > 0:
                idx = valid_indices[-1]
                return thresholds[idx]
            else:
                return max(thresholds)

    def _calculate_metrics(
        self, y_true, y_pred_proba, medium_threshold, high_threshold
    ):
        """Calculate FNR, FPR and classification percentages."""
        y_pred_medium = (y_pred_proba >= medium_threshold).astype(int)
        y_pred_high = (y_pred_proba >= high_threshold).astype(int)

        tn_medium = np.sum((y_true == 0) & (y_pred_medium == 0))
        fp_medium = np.sum((y_true == 0) & (y_pred_medium == 1))
        fn_medium = np.sum((y_true == 1) & (y_pred_medium == 0))
        tp_medium = np.sum((y_true == 1) & (y_pred_medium == 1))

        tn_high = np.sum((y_true == 0) & (y_pred_high == 0))
        fp_high = np.sum((y_true == 0) & (y_pred_high == 1))
        fn_high = np.sum((y_true == 1) & (y_pred_high == 0))
        tp_high = np.sum((y_true == 1) & (y_pred_high == 1))

        fnr_medium = (
            fn_medium / (fn_medium + tp_medium) if (fn_medium + tp_medium) > 0 else 0
        )
        fpr_medium = (
            fp_medium / (fp_medium + tn_medium) if (fp_medium + tn_medium) > 0 else 0
        )
        fnr_high = fn_high / (fn_high + tp_high) if (fn_high + tp_high) > 0 else 0
        fpr_high = fp_high / (fp_high + tn_high) if (fp_high + tn_high) > 0 else 0

        y_classified = np.zeros_like(y_true, dtype=int) - 1  # -1 for gray area
        y_classified[y_pred_proba < medium_threshold] = 0  # Negative
        y_classified[y_pred_proba >= high_threshold] = 1  # Positive

        neg_percent = np.mean(y_classified == 0)
        pos_percent = np.mean(y_classified == 1)

        return {
            "fnr_medium": fnr_medium,
            "fpr_medium": fpr_medium,
            "fnr_high": fnr_high,
            "fpr_high": fpr_high,
            "neg_percent": neg_percent,
            "pos_percent": pos_percent,
        }

    def _get_positive_class_proba(self, model, X):
        """Extract probability for the positive class."""
        proba = model.predict_proba(X)
        if proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()

    def evaluate_with_super_classifier(self, X, y):
        """Evaluate models using SuperClassifier with repeated stratified k-fold cross-validation."""

        testing_config = TestingConfig(
            models=[self.baseline_config, self.target_config],
            random_state=self.random_state,
            verbose=True,
        )

        super_clf = SuperClassifier(testing_config)

        super_clf.fit_evaluate(X, y)

        rskf = RepeatedStratifiedKFold(
            n_splits=5, n_repeats=3, random_state=self.random_state
        )
        baseline_model = super_clf.trained_models["Baseline"]
        target_model = super_clf.trained_models["Target"]

        for train_idx, test_idx in rskf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            baseline_proba = self._get_positive_class_proba(baseline_model, X_test)

            baseline_auroc = roc_auc_score(y_test, baseline_proba)
            baseline_auprc = average_precision_score(y_test, baseline_proba)

            baseline_train_proba = self._get_positive_class_proba(
                baseline_model, X_train
            )
            baseline_medium_threshold = self._find_threshold(
                y_train, baseline_train_proba, "sensitivity", 0.99
            )
            baseline_high_threshold = self._find_threshold(
                y_train, baseline_train_proba, "specificity", 0.90
            )

            baseline_metrics = self._calculate_metrics(
                y_test,
                baseline_proba,
                baseline_medium_threshold,
                baseline_high_threshold,
            )

            self.results["baseline"]["auprc"].append(baseline_auprc)
            self.results["baseline"]["auroc"].append(baseline_auroc)
            for key, value in baseline_metrics.items():
                self.results["baseline"][key].append(value)

            target_proba = self._get_positive_class_proba(target_model, X_test)

            target_auroc = roc_auc_score(y_test, target_proba)
            target_auprc = average_precision_score(y_test, target_proba)

            target_train_proba = self._get_positive_class_proba(target_model, X_train)
            target_medium_threshold = self._find_threshold(
                y_train, target_train_proba, "sensitivity", 0.99
            )
            target_high_threshold = self._find_threshold(
                y_train, target_train_proba, "specificity", 0.90
            )

            target_metrics = self._calculate_metrics(
                y_test, target_proba, target_medium_threshold, target_high_threshold
            )

            self.results["target"]["auprc"].append(target_auprc)
            self.results["target"]["auroc"].append(target_auroc)
            for key, value in target_metrics.items():
                self.results["target"][key].append(value)

        return super_clf

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Calculate average metrics across all iterations."""
        summary = {}
        for model_name in ["baseline", "target"]:
            summary[model_name] = {}
            for metric in self.results[model_name]:
                summary[model_name][metric] = np.mean(self.results[model_name][metric])
        return summary

    def plot_results(self, name, figsize=(10, 12)):
        """Plot comparison of baseline and target models with better spacing."""
        summary = self.get_summary()
        metrics = [
            ("auprc", "AUPRC"),
            ("auroc", "AUROC"),
            ("fnr_medium", "FNR (Medium Risk)"),
            ("fpr_medium", "FPR (Medium Risk)"),
            ("fnr_high", "FNR (High Risk)"),
            ("fpr_high", "FPR (High Risk)"),
            ("neg_percent", "Negative Class %"),
            ("pos_percent", "Positive Class %"),
        ]

        fig, axes = plt.subplots(4, 2, figsize=figsize)
        axes = axes.flatten()
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        for i, (metric_key, metric_name) in enumerate(metrics):
            baseline_val = summary["baseline"][metric_key]
            target_val = summary["target"][metric_key]

            ax = axes[i]
            bars = ax.bar(
                ["Baseline", "Target"],
                [baseline_val, target_val],
                color=["skyblue", "lightgreen"],
                width=0.5,
            )

            for bar in bars:
                height = bar.get_height()

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.4f}",
                    ha="center",
                    va="top",
                    fontsize=11,
                )

            ax.set_title(metric_name, fontsize=13, fontweight="bold", pad=10)
            ax.set_ylim(0, min(1.1, max(baseline_val, target_val) * 1.2))

            if "percent" in metric_key:
                ax.set_ylim(0, 1.0)
                # ax.set_ylabel("Percentage")

        plt.savefig(f"{name}.png", dpi=300)
        plt.show()
