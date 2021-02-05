from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np


class TreeFeature:
    def __init__(self, cols=None, tree_model=None, keep_cols=False):
        self.cols = cols
        self.keep_cols = keep_cols
        self.tree_model = tree_model
        self.ohe = OneHotEncoder()

    def fit(self, X, y):
        n = X.shape[0]
        ind = np.arange(n)
        np.random.shuffle(ind)
        ind = ind[: int(0.3 * n)]
        X = X.iloc[ind, :]
        y = y[ind]

        if self.cols is None:
            self.cols = X.columns

        self.tree_model.fit(X[self.cols], y)
        tree_feature = self.tree_model.apply(X[self.cols])
        if tree_feature.ndim > 2:
            tree_feature = tree_feature[:, :, 0]
        self.ohe.fit(tree_feature)

    def transform(self, X, **kwargs):

        tree_feature = self.tree_model.apply(X[self.cols])
        if tree_feature.ndim > 2:
            tree_feature = tree_feature[:, :, 0]

        tree_feature = self.ohe.transform(tree_feature)

        tree_cols = [f"tree_feature_{i}" for i in range(tree_feature.shape[1])]

        tree_feature = tree_feature.toarray()
        tree_feature = pd.DataFrame(tree_feature, columns=tree_cols)

        if self.keep_cols:
            cols_to_keep = list(X.columns) + tree_cols
        else:
            cols_to_keep = [
                col for col in X.columns if col not in self.cols
            ] + tree_cols

        X_tr = pd.concat([X.reset_index(drop=True), tree_feature], axis=1)

        return X_tr[cols_to_keep]


def polynomial_feature_name(data, cols, d=2, interaction_only=True):
    """
    MinMaxNormalization( polynomial_feature_name(data, cols=cols_poly) )
    """
    x = data.sample(1)
    x.fillna(0, inplace=True)
    x_tr = _polynomial_features(x, cols, d=d, interaction_only=interaction_only)

    return [col for col in x_tr.columns if col not in x.columns] + cols


class IdTransform:
    """"""

    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        pass

    def transform(self, X, **kwargs):
        return X


def _polynomial_features(df, cols, d=2, interaction_only=False):
    if cols is None:
        cols = df.columns()

    X = df[cols].copy()
    poly = PolynomialFeatures(d, include_bias=False, interaction_only=interaction_only)
    X_tr = poly.fit_transform(X)

    input_cols = list(X.columns)
    output_cols = poly.get_feature_names(input_features=input_cols)

    X_tr = pd.DataFrame(X_tr, columns=output_cols).drop(input_cols, axis=1)

    return pd.concat([df.reset_index(drop=True), X_tr], axis=1)


def polynomial_features(d=2, interaction_only=False):
    return lambda df, cols, **kwargs: _polynomial_features(
        df, cols, d=d, interaction_only=interaction_only
    )


class PolynomialTransform:
    def __init__(self, cols, d=2, interaction_only=False, norm_strategy="minmax"):
        self.cols = cols
        self.d = d
        self.interaction_only = interaction_only
        self.norm_strategy = norm_strategy
        self.normalizer = None

    def fit(self, X, y):
        pass


class StaticTransform:
    """
    Exemple
    -------
    poly_tr = StaticTransform(["a", "b"], polynomial_features)
    poly_tr.transform(df, d=3, interaction_only=True)
    """

    def __init__(self, cols, func):
        self.cols = cols
        self.func = func

    def fit(self, X=None, y=None):
        pass

    def transform(self, X, **kwargs):
        return self.func(X, self.cols, **kwargs)


class ComposeTransforms:
    """
    list_tr = [
              ImputeWithMean(["a", "b"]),
              StaticTransform(["a", "b"], polynomial_features)
    ]
    tr = ComposeTransforms( list_tr )
    tr.fit(df)
    tr.transform(df)

    """

    def __init__(self, list_transforms: list):
        self.transforms = list_transforms

    def fit(self, X, y=None, **kwargs):
        kwargs["is_training"] = True

        new_X = X.copy()

        for transform in self.transforms:
            transform.fit(new_X, y)
            new_X = transform.transform(new_X, **kwargs)

    def transform(self, X, **kwargs):
        new_X = X.copy()

        for transform in self.transforms:
            new_X = transform.transform(new_X, **kwargs)

        return new_X


class Imputers:
    """
    Imputers( [
      ImputerMean(['col1', 'col3']),
      ImputerMode(['col2']),
     ] )
    """

    def __init__(self, imputers: list):
        self.imputers = imputers

    def fit(self, X, y=None):
        for imputer in self.imputers:
            imputer.fit(X, y)

    def transform(self, X):
        new_X = X.copy()
        for imputer in self.imputers:
            new_X = imputer.transform(new_X)
        return new_X


class One_Hot_Encoder:

    """"""

    def __init__(self, cols: list):
        self.encoder = OneHotEncoder(handle_unknown="error", sparse=False)
        self.cols = cols

    def fit(self, X, y=None, **kwargs):
        X_new = X[self.cols].copy()
        self.encoder.fit(X_new)

    def transform(self, X, **kwargs):
        X_new = X.copy()
        X_cat = X_new[self.cols].copy()

        vals = self.encoder.transform(X_cat)
        cols = []

        for i, col in enumerate(X_cat.columns):
            for unique in self.encoder.categories_[i]:
                cols.append(f"{col}_{unique}")

        X_tr = pd.DataFrame(data=vals, columns=cols, index=X_cat.index).astype(int)

        X_new = X_new.drop(columns=self.cols)

        return pd.concat([X_new, X_tr], axis=1)


class Ordinal_Encoder:

    """"""

    def __init__(self, cols: list):
        self.encoder = OrdinalEncoder()
        self.cols = cols

    def fit(self, X, y=None, **kwargs):
        X_new = X[self.cols].copy()
        self.encoder.fit(X_new)

    def transform(self, X, **kwargs):
        X_new = X.copy()
        X_cat = X_new[self.cols].copy()

        vals = self.encoder.transform(X_cat)
        cols = X_cat.columns

        X_tr = pd.DataFrame(data=vals, columns=cols, index=X_cat.index).astype(int)

        X_new = X_new.drop(columns=self.cols)

        return pd.concat([X_new, X_tr], axis=1)
