import numpy as np
import pandas as pd
import re

from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from num2words import num2words
from sklearn.impute import SimpleImputer


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.feature_names]


class DictionaryVectorizer(BaseEstimator, TransformerMixin):
    """
    1. Extract values from dictionaries within records
    2. Join them in one string
    3. Vectorize it / dummify it - CountVectorizer
    """

    def __init__(self, key, all_=True):  # all_ ?
        self.key = key
        self.all = all_

    def fit(self, X, y=None):
        genres = X.apply(
            lambda x: self.extract_items(x, self.key, self.all)
        )  # names of values that correspond to key in dict
        self.vectorizer = CountVectorizer().fit(genres)  # BOW
        self.columns = self.vectorizer.get_feature_names_out()
        return self

    def transform(self, X):
        genres = X.apply(lambda x: self.extract_items(x, self.key))  # why not all
        data = self.vectorizer.transform(genres)
        return pd.DataFrame(
            data.toarray(),
            columns=self.vectorizer.get_feature_names_out(),
            index=X.index,
        )

    @staticmethod
    def extract_items(list_, key, all_=True):
        sub = lambda x: re.sub(r"[^A-Za-z0-9]", "_", x)
        if all_:
            target = []
            for dict_ in eval(list_):
                target.append(sub(dict_[key].strip()))
            return " ".join(target)
        elif not eval(list_):  # ?
            return "no_data"
        else:
            return sub(eval(list_)[0][key].strip())


# Choose most popular DUMMY features
class TopFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, percent=None, quantify=None):
        if percent is not None:
            if percent > 100:
                self.percent = 100
            else:
                self.percent = percent
            self.quantify = None
        elif quantify is not None:
            self.percent = None
            self.quantify = quantify

    def fit(self, X, y=None):
        counts = X.sum().sort_values(ascending=False)
        if self.percent is not None:
            index_ = int(counts.shape[0] * self.percent / 100)
        elif self.quantify is not None:
            index_ = self.quantify
        self.columns = counts[:index_].index
        return self

    def transform(self, X):
        return X[self.columns]


# Sum across given features
class SumTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, series_name):
        self.series_name = series_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.sum(axis=1).to_frame(self.series_name)


# Takes input function that decides whether to label column True or False
class Binarizer(BaseEstimator, TransformerMixin):
    def __init__(self, condition, name):
        self.condition = condition
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda x: int(self.condition(x))).to_frame(self.name)


# Date utils
def get_year(date):
    return datetime.strptime(date, "%Y-%m-%d").year


def get_month_name(date):
    return datetime.strptime(date, "%Y-%m-%d").strftime("%b")


def get_weekday_name(date):
    return datetime.strptime(date, "%Y-%m-%d").strftime("%a")


def get_month(date):
    return datetime.strptime(date, "%Y-%m-%d").month


def get_weekday(date):
    return datetime.strptime(date, "%Y-%m-%d").day


# Date extraction and encoding
class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, encoding="both"):
        self.encoding = encoding

    def fit(self, X, y=None):
        return self

    @staticmethod
    def one_hot_encoding(X):
        month = pd.get_dummies(X.apply(get_month_name))
        day = pd.get_dummies(X.apply(get_weekday_name))
        return pd.concat([month, day], axis=1)

    @staticmethod
    def sin_cos_encoding(X):
        encoded = pd.DataFrame()
        encoded["month_sin"] = X.apply(lambda x: np.sin(2 * np.pi * get_month(x) / 12))
        encoded["month_cos"] = X.apply(lambda x: np.cos(2 * np.pi * get_month(x) / 12))
        encoded["day_sin"] = X.apply(lambda x: np.sin(2 * np.pi * get_weekday(x) / 7))
        encoded["day_cos"] = X.apply(lambda x: np.cos(2 * np.pi * get_weekday(x) / 7))
        return encoded

    def transform(self, X):
        year = X.apply(get_year).rename("year")  # series
        if self.encoding == "one_hot":
            encoded = self.one_hot_encoding(X)
        elif self.encoding == "sin_cos":
            encoded = self.sin_cos_encoding(X)
        else:  # both
            encoded = pd.concat(
                [self.one_hot_encoding(X), self.sin_cos_encoding(X)], axis=1
            )
        return pd.concat([year, encoded], axis=1)


class ItemCounter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda x: int(self.get_list_len(x)))

    @staticmethod
    def get_list_len(list_):
        return len(eval(list_))


class MeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.mean(axis=1).to_frame(self.name)


class AgeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.word_to_num = {num2words(i): i for i in range(150)}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def convert_to_int(x):
            try:
                return int(x)
            except ValueError:
                return self.word_to_num[x.lower()]

        return X.apply(lambda x: convert_to_int(x))


class HeightWeightHandler(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.possible_splits = [",", "_", "-", ";"]
        self.word_to_num = {num2words(i): i for i in range(300)}
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def splitter(x):
            for char in x:
                if char in self.possible_splits:
                    height, weight = x.split(char, 1)
                    return height, weight
            return None

        def convert_to_int(x):
            try:
                return int(x)
            except ValueError:
                return self.word_to_num[x.lower()]

        def handler(x):
            height, weight = splitter(x)
            weight = convert_to_int(weight)
            height = float(height)
            return pd.Series({"height": height, "weight": weight})

        return X[self.column_name].apply(handler)


# Probalby not needed
class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, strategy):
        self.columns = columns
        self.imputer = SimpleImputer(strategy=strategy)

    def fit(self, X, y=None):
        self.imputer.fit(X[self.columns])
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.columns] = self.imputer.transform(X[self.columns])
        return X_


class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, scaler):
        self.columns = columns
        self.scaler = scaler

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.columns] = self.scaler.transform(X[self.columns])
        return X_
