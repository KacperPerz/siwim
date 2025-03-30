import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.base import BaseEstimator
import importlib
from typing import Dict, Any, Optional, Tuple, Union
import logging
from config import TestingConfig, ModelConfig, EnsembleConfig, EnsembleEnum
from sklearn.metrics import make_scorer
from config import CV


class SuperClassifier:
    def __init__(self, config: TestingConfig):
        self.config = config
        self.results = pd.DataFrame()
        self.best_model_name = None
        self.best_score = -np.inf
        self.trained_models: dict[str, BaseEstimator] = {}
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("SuperClassifier")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        return logger

    def _get_estimator(self, model_config: ModelConfig) -> Any:
        """Dynamically import and instantiate the estimator."""
        try:
            if isinstance(model_config.estimator_class, str):
                parts = model_config.estimator_class.split(".")
                module_name = ".".join(parts[:-1])
                class_name = parts[-1]
                module = importlib.import_module(module_name)
                estimator_class = getattr(module, class_name)
                return estimator_class(**model_config.params)
            else:
                return model_config.estimator_class(**model_config.params)
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Failed to import {model_config.estimator_class}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error instantiating {model_config.name}: {e}")
            raise

    def _create_ensemble(
        self, ensemble_config: EnsembleConfig
    ) -> Union[VotingClassifier, StackingClassifier]:
        """Create stacking or voting ensemble based on config."""
        if not self.config.ensemble_models:
            raise ValueError("No ensemble models defined in configuration")

        estimators = []
        for estimator in self.config.ensemble_models:
            estimators.append((estimator.__class__.__name__, estimator))

        if ensemble_config.ensemble_type == EnsembleEnum.voting:
            return VotingClassifier(
                estimators=estimators,
                voting=ensemble_config.voting_type,
                weights=ensemble_config.weights,
            )
        elif ensemble_config.ensemble_type == EnsembleEnum.stacking:
            # meta-learner for stacking
            if isinstance(ensemble_config.final_estimator, dict):
                # dict config
                meta_config = ModelConfig(**ensemble_config.final_estimator)
                final_estimator = self._get_estimator(meta_config)
            else:
                # direct estimator obj
                final_estimator = ensemble_config.final_estimator

            return StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=CV,
            )
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_config.ensemble_type}")

    def fit_evaluate(self, X, y) -> pd.DataFrame:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )

        results = []

        for model_config in self.config.models:
            self.logger.info(f"Evaluating {model_config.name}...")

            try:
                # handle ensambles
                self.logger.info("Before creating ensemble")
                if (
                    model_config.ensemble_config
                    and model_config.ensemble_config.ensemble_type != EnsembleEnum.none
                ):
                    model = self._create_ensemble(model_config.ensemble_config)
                else:
                    model = self._get_estimator(model_config)

                scorer = make_scorer(
                    model_config.scoring.func, **model_config.scoring.kwargs
                )
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=model_config.cv, scoring=scorer
                )

                model.fit(X_train, y_train)
                self.trained_models[model_config.name] = model
                y_pred = model.predict(X_test)
                test_score = model_config.scoring(y_test, y_pred)
                if test_score > self.best_score:
                    self.best_score = test_score
                    self.best_model_name = model_config.name

                results.append(
                    {
                        "name": model_config.name,
                        "is_ensemble": model_config.is_ensemble,
                        "cv_mean": cv_scores.mean(),
                        "cv_std": cv_scores.std(),
                        "test_score": test_score,
                    }
                )

                self.logger.info(
                    f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
                )
                self.logger.info(f"  Test Score: {test_score:.4f}")

            except Exception as e:
                self.logger.error(f"Error evaluating {model_config.name}: {e}")
                continue

        self.results = pd.DataFrame(results)
        return self.results

    def plot_results(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        if self.results.empty:
            raise ValueError("No results to plot. Run fit_evaluate first.")

        fig, ax = plt.subplots(figsize=figsize)
        self.results.plot(
            x="name",
            y="cv_mean",
            kind="bar",
            yerr="cv_std",
            ax=ax,
            alpha=0.7,
            color="skyblue",
            label="CV Score",
        )
        self.results.plot(
            x="name",
            y="test_score",
            kind="line",
            marker="o",
            color="red",
            ax=ax,
            label="Test Score",
        )
        ax.set_title("Model Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_xticklabels(self.results["name"], rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        return fig

    def get_best_model(self) -> Dict[str, Any]:
        if not self.best_model_name:
            raise ValueError("No models evaluated yet. Run fit_evaluate first.")

        return {
            "name": self.best_model_name,
            "model": self.trained_models[self.best_model_name],
            "score": self.best_score,
        }

    def predict(self, X, model_name: Optional[str] = None):
        if not self.trained_models:
            raise ValueError("No trained models available. Run fit_evaluate first.")

        if model_name is None:
            if not self.best_model_name:
                raise ValueError("No best model determined. Run fit_evaluate first.")
            model_name = self.best_model_name

        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models.")

        return self.trained_models[model_name].predict(X)

    def predict_proba(self, X, model_name: Optional[str] = None):
        if not self.trained_models:
            raise ValueError("No trained models available. Run fit_evaluate first.")

        if model_name is None:
            if not self.best_model_name:
                raise ValueError("No best model determined. Run fit_evaluate first.")
            model_name = self.best_model_name

        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models.")

        model = self.trained_models[model_name]

        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                f"Model '{model_name}' does not support probability predictions."
            )

        return model.predict_proba(X)
