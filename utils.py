import pandas as pd
import numpy as np
import logging

from typing import Any, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
)
from evaluator import ClassifierEvaluator
from super_classifier import SuperClassifier
from config import ModelConfig, RANDOM_STATE


def list_unique_object_values(df: pd.DataFrame) -> None:
    for c in [_ for _ in df.columns if df[_].dtype == "object"]:
        print(c, df[c].unique())


def clean_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    return df


def convert_to_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def missing_values_stats(X: pd.DataFrame) -> None:
    """
    Print the percentage of missing values in each column of the DataFrame.
    """
    missing_values = X.isnull().sum()
    missing_percentage = (missing_values / len(X)) * 100
    missing_info = pd.DataFrame(
        {"Missing Values": missing_values, "Percentage": missing_percentage}
    )
    print(
        missing_info[missing_info["Missing Values"] > 0].sort_values(
            "Percentage", ascending=False
        )
    )


def binarize_decision_class(y, positive_classes=[1, 2]):  # discretize the class
    if isinstance(y, pd.Series):
        return y.isin(positive_classes).astype(int)
    else:
        return pd.Series(np.isin(y, positive_classes).astype(int))


def preprocess_medical_data(
    df: pd.DataFrame,
    target_column: str,
    num_features: list[str],
    onehot_features: list[str],
    ordinal_features: list[str],
    yes_no_features: list[str],
    drop_features: list[str],
    keep_features: list[str],
    cat_features: list[str],
    additional_features: Optional[
        list[tuple[list[str], dict[str, Any]]]
    ] = None,  # additonal features with dict of parameters to map - with replace method
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X = convert_to_numeric(X, num_features)
    X = clean_object_columns(X)
    X = X.replace("?", None)

    for feature in yes_no_features:
        if feature in X.columns:
            X[feature] = X[feature].replace(
                {
                    "YES": 1,
                    "NO": 0,
                    None: np.nan,
                    "None": np.nan,
                    "Yes": 1,
                    "No": 0,
                }
            )

    for feature_config in additional_features or []:
        features, mapping = feature_config
        for feature in features:
            if feature in X.columns:
                X[feature] = X[feature].replace(mapping)

    X = X.drop(columns=[col for col in drop_features if col in X.columns])

    cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing")
    num_imputer = SimpleImputer(strategy="median")

    existing_cat_features = [col for col in cat_features if col in X.columns]
    existing_num_features = [col for col in num_features if col in X.columns]

    if existing_cat_features:
        X[existing_cat_features] = cat_imputer.fit_transform(X[existing_cat_features])

    if existing_num_features:
        X[existing_num_features] = num_imputer.fit_transform(X[existing_num_features])

    existing_onehot_features = [col for col in onehot_features if col in X.columns]
    existing_ordinal_features = [col for col in ordinal_features if col in X.columns]
    existing_keep_features = [col for col in keep_features if col in X.columns]

    for col in existing_onehot_features:
        X[col] = X[col].astype(str)

    for col in existing_ordinal_features:
        X[col] = X[col].astype(str)

    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )

    encoded_dfs = []

    if existing_num_features:
        encoded_dfs.append(X[existing_num_features])

    if existing_onehot_features:
        onehot_encoded = onehot_encoder.fit_transform(X[existing_onehot_features])
        onehot_columns = onehot_encoder.get_feature_names_out(existing_onehot_features)
        encoded_dfs.append(
            pd.DataFrame(onehot_encoded, columns=onehot_columns, index=X.index)
        )

    if existing_ordinal_features:
        ordinal_encoded = ordinal_encoder.fit_transform(X[existing_ordinal_features])
        ordinal_columns = [f"{feat}_encoded" for feat in existing_ordinal_features]
        encoded_dfs.append(
            pd.DataFrame(ordinal_encoded, columns=ordinal_columns, index=X.index)
        )

    if existing_keep_features:
        encoded_dfs.append(X[existing_keep_features])

    X_encoded = pd.concat(encoded_dfs, axis=1)

    for col in X_encoded.columns:
        X_encoded[col] = X_encoded[col].astype(float)

    return X_encoded, y


def full_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    baseline_config: ModelConfig,
    target_config: ModelConfig,
    target_name: str,
) -> tuple[ClassifierEvaluator, SuperClassifier]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("ClassifierEvaluation")

    y_binary = binarize_decision_class(y)
    evaluator = ClassifierEvaluator(baseline_config, target_config, RANDOM_STATE)
    super_clf = evaluator.evaluate_with_super_classifier(X, y_binary)
    summary = evaluator.get_summary()

    logger.info("Evaluation Results Summary:")
    for model_name, metrics in summary.items():
        logger.info(f"\n{model_name.upper()} MODEL:")
        for metric_name, value in metrics.items():
            logger.info(f" {metric_name}: {value:.4f}")

    evaluator.plot_results(target_name)
    logger.info(f"Results plot saved as '{target_name}.png'")

    return evaluator, super_clf
