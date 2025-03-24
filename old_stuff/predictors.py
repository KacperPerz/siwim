import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.base import BaseEstimator
from typing import Dict, List, Type, Union, Tuple, Optional, Callable

# Should I consider using optuna here or it is overkill?

# Scoring: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter


def cv_model(
    cv: int,
    rand: bool,
    model: Union[Type[BaseEstimator], BaseEstimator],
    params: Dict,
    scoring: Optional[Callable] = None,
):
    if rand:
        search = RandomizedSearchCV(
            model, param_distributions=params, cv=cv, scoring=scoring
        )
    else:
        search = GridSearchCV(model, param_grid=params, cv=cv, scoring=scoring)
    return search


def stacking_model_pipe(
    X: pd.DataFrame,
    y: pd.DataFrame,
    models: List[Union[Type[BaseEstimator], BaseEstimator]],
    params: List[Dict],
    final_estimator: BaseEstimator,
    cv: int = 10,
    model_type: str = "regression",
    scoring: Optional[Callable] = None,
) -> Union[
    Tuple[StackingRegressor, List[Tuple[str, BaseEstimator]]],
    Tuple[StackingClassifier, List[Tuple[str, BaseEstimator]]],
]:
    if scoring is None:  # Default scoring
        scoring = (
            "f1_weighted"
            if model_type == "classification"
            else "neg_mean_absolute_error"
        )

    ready_models = []
    for model, p in zip(models, params):
        ready_models.append(
            cv_model(cv=cv, rand=True, model=model, params=p, scoring=scoring)
        )

    for i in range(len(ready_models)):
        ready_models[i].fit(X, y)

    estimators = [(str(m), m.best_estimator_) for m in ready_models]

    if model_type == "regression":
        ensemble = StackingRegressor(estimators, final_estimator=final_estimator)
    elif model_type == "classification":
        ensemble = StackingClassifier(estimators, final_estimator=final_estimator)
    else:
        raise ValueError("Invalid model_type. Choose 'regression' or 'classification'.")

    ensemble.fit(X, y)

    return ensemble, estimators


def plot_weak_learners_feature_importance(
    feature_importances: pd.DataFrame, nlargest: int = 10
):
    for i in range(len(feature_importances)):
        plt.figure(figsize=(5, 5))
        pd.Series(
            feature_importances.iloc[i], index=feature_importances.columns
        ).nlargest(nlargest).plot(kind="barh")
        plt.show()
