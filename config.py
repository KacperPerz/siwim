from typing import Dict, Any, List, Optional, Callable, Union
from numpy import ndarray
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import StrEnum
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics._classification import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    class_likelihood_ratios,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    hamming_loss,
    hinge_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    zero_one_loss,
)

TEST_SIZE = 0.2
RANDOM_STATE = 42
VERBOSE = False
CV = 5

ALLOWED_SCORING_FUNCTIONS = {
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    class_likelihood_ratios,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    hamming_loss,
    hinge_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    zero_one_loss,
}


class EnsembleEnum(StrEnum):
    """Enum for ensemble types"""

    none = "none"
    stacking = "stacking"
    voting = "voting"


class VotingEnum(StrEnum):
    """Enum for voting types"""

    hard = "hard"
    soft = "soft"


class ScoringConfig(BaseModel):
    func: Callable[..., float | ndarray] = accuracy_score
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __call__(
        self, y_true: float | ndarray, y_pred: float | ndarray
    ) -> float | ndarray:
        return self.func(y_true, y_pred, **self.kwargs)

    @field_validator("func")
    def check_scoring_function(
        cls: Callable[..., float | ndarray], v
    ) -> Callable[..., float | ndarray]:
        if v not in ALLOWED_SCORING_FUNCTIONS:
            raise ValueError(
                f"Invalid scoring function {v}. Allowed functions are: {ALLOWED_SCORING_FUNCTIONS}"
            )
        return v


class EnsembleConfig(BaseModel):
    """Configuration for ensemble models"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ensemble_type: EnsembleEnum = EnsembleEnum.none
    voting_type: VotingEnum = VotingEnum.hard
    weights: Optional[List[float]] = None
    final_estimator: Union[Dict[str, Any], BaseEstimator] = (
        LogisticRegression()
    )  # meta-learner config; default is LogisticRegression


class ModelConfig(BaseModel):
    """Configuration for individual models"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    estimator_class: Any
    is_ensemble: bool = False
    params: Dict[str, Any] = Field(default_factory=dict)
    cv: int = CV
    scoring: ScoringConfig = ScoringConfig(func=accuracy_score)
    ensemble_config: Optional[EnsembleConfig] = None


class TestingConfig(BaseModel):
    """Configuration for model testing"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    models: List[ModelConfig]
    ensemble_models: List[BaseEstimator] = Field(default_factory=list)
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    verbose: bool = VERBOSE
