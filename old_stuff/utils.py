import random
import pandas as pd
from num2words import num2words
from scipy.stats import truncnorm
from typing import Tuple


def handle_same_name_col(df: pd.DataFrame) -> pd.DataFrame:
    column_names = set()
    for i, column in enumerate(df.columns):
        if column in column_names:
            print("Duplicate found")
            df.columns.values[i] = f"{column}_{i}"
        else:
            column_names.add(column)
    return df


def data_with_important_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_importances: pd.DataFrame,
    conjunction=True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Experimental:
    Selects important features based on their positive importance values across multiple weak learners.

    If conjunction is set to True, only features with positive importance that occur in ALL weak learners
    are designated as chosen columns. Otherwise, any feature with positive importance in at least one
    weak learner is selected.
    """
    if conjunction:
        important_columns = set(
            feature_importances.columns[feature_importances.iloc[0] > 0]
        )
        for _, row in feature_importances.iterrows():
            columns_satisfying_condition = set(feature_importances.columns[row > 0])
            important_columns.intersection_update(columns_satisfying_condition)
    else:
        important_columns = set()
        for _, row in feature_importances.iterrows():
            col_satisfying_condition = feature_importances.columns[row > 0]
            important_columns.update(col_satisfying_condition)

    important_columns = list(important_columns)
    return X_train[important_columns], X_test[important_columns]


# Noisy data generation util functions
def generate_age():
    choice = random.choice(["integer", "word"])
    if choice == "integer":
        return random.randint(1, 100)  # Generate random integer age
    elif choice == "word":
        return num2words(random.randint(1, 100))


def generate_height_weight():
    a = (1.3 - 1.7) / 0.2
    b = (2.3 - 1.7) / 0.2
    choice = random.choice([";", ",", "-", "_"])
    words = random.choice([True, False])
    height = truncnorm.rvs(a, b, loc=1.7, scale=0.2, size=1)[0]
    weight = random.randrange(40, 150)
    if words:
        weight = num2words(weight)
    return str(height) + choice + str(weight)


def generate_age_column(n=100):
    return pd.Series([generate_age() for _ in range(n)])


def generate_height_weight_column(n=100):
    return pd.Series([generate_height_weight() for _ in range(n)])


def generate_categorical_column(categories, n=100):
    return pd.Series([random.choice(categories) for _ in range(n)])


def generate_missing_numeric_column(minv=1000, maxv=10000, n=100):
    return pd.Series(
        [random.choice([random.randint(minv, maxv), None]) for _ in range(n)]
    )


def generate_yes_no_column(n=100):
    return pd.Series([random.choice(["yes", "no"]) for _ in range(n)])
