import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.lower_ = X.quantile(self.lower_quantile)
        self.upper_ = X.quantile(self.upper_quantile)
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_", "upper_"])
        X_winsorized = X.clip(lower=self.lower_, upper=self.upper_, axis=1)
        return X_winsorized
