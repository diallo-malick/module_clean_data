import pandas as pd
import numpy as np
from sklearn.preprocessing import power_transform


def replace_by_const(df: pd.DataFrame, col_const: dict):
    """
    Remplace les valeurs manquantes par une constante
    df: pd.DataFrame
    col_const: dict

    Exemple:
    col_const = {
      "colName1": -9999
      "colName2": 0
      "colName3": 1000
    }
    replace_by_const(df, col_const)
    """
    data = df.copy()
    data.fillna(value=col_const, inplace=True)

    return data


class ImputeWithMean:
    """
    Impute par la moyenne

    """

    def __init__(self, cols=None):
        self.means = {col: None for col in cols}

    def fit(
        self,
        X,
        y=None,
    ):
        cols = self.means.keys()
        self.means = {col: X[col].mean() for col in cols}

    def transform(self, X: pd.DataFrame, **kwargs):
        X = X.copy()
        X.fillna(value=self.means, inplace=True)
        return X


class ImputeWithMedian:
    """
    Impute par la mediane

    """

    def __init__(self, cols=None):
        self.medians = {col: None for col in cols}

    def fit(self, X, y=None):
        cols = self.medians.keys()
        self.medians = {col: X[col].median() for col in cols}

    def transform(self, X: pd.DataFrame, **kwargs):
        X = X.copy()
        X.fillna(value=self.medians, inplace=True)
        return X


class ImputeWithMode:
    """
    Impute par le mode

    """

    def __init__(self, cols=None):
        self.modes = {col: None for col in cols}

    def fit(self, X, y=None):
        X = X.copy()
        cols = self.modes.keys()
        self.modes = {col: X[col].mode() for col in cols}

    def transform(self, X: pd.DataFrame, **kwargs):
        X = X.copy()
        X.fillna(value=self.modes, inplace=True)
        return X


class ImputeOutOfRange:
    def __init__(self, cols=None, is_numeric=True):
        self.is_numeric = is_numeric
        self.unk = {col: None for col in cols}

    def fit(self, X, y=None):
        UNK_TOKEN = "UNK"
        cols = self.unk.keys()

        if self.is_numeric:
            self.unk = {col: self.get_out_of_range(X[col]) for col in cols}
        else:
            self.unk = {col: UNK_TOKEN for col in cols}

    def transform(self, X: pd.DataFrame, **kwargs):
        X = X.copy()
        X.fillna(value=self.unk, inplace=True)
        return X

    def get_out_of_range(self, col_values):
        a = col_values.min()
        nb_chiffre = 1 + int(np.log10(a))
        return int((nb_chiffre + 1) * "1") * 9


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

    def transform(self, X, **kwargs):
        new_X = X.copy()
        for imputer in self.imputers:
            new_X = imputer.transform(new_X)
        return new_X


def impute_w_inst_imputers(df: pd.DataFrame, imputers: list) -> pd.DataFrame:

    """
    impute chaque variables(colonnes) avec la liste d'imputers donnée

    ------------
    df: Dataset avec valeur manquante
    imputers:  Liste d'instances des des classes des imputers definient plus haut qui a déjà été fité
    Ex: df = pd.DataFrame({'a':[1,1,1, np.nan, np.nan], 'b': [2,2,2, np.nan, np.nan]})
        imputer1 = ImputeWithMean(['a'])
        imputer2 = ImputeWithMedian(['b'])
        imputers = [imputer1, imputer2]
        imputer1.fit(df)
        imputer2.fit(df)

        df.values = array([[1., 2.],
                        [1., 2.],
                        [1., 2.],
                        [1., 2.],
                        [1., 2.]])
    --------
    Ex:
    """

    for imp in imputers:
        imp.transform(df)
    return df


### NORMALIZATION


class MinMaxNormalization:
    """
    Impute par la mediane

    """

    def __init__(self, cols=None):
        self.cols = cols
        self.min = {col: None for col in self.cols}
        self.max = {col: None for col in self.cols}

    def fit(self, X, y=None):
        cols = self.min.keys()
        self.min = {col: X[col].min() for col in self.cols}
        self.max = {col: X[col].max() for col in self.cols}

    def transform(self, X: pd.DataFrame, **kwargs):
        X = X.copy()

        for col in self.cols:
            X[col] = (X[col] - self.min[col]) / (self.max[col] - self.min[col])

        return X


class StdNormalization:
    """
    Impute par la mediane

    """

    def __init__(self, cols=None):
        self.cols = cols
        self.mean = {col: None for col in self.cols}
        self.std = {col: None for col in self.cols}

    def fit(self, X, y=None):
        # cols = self.min.keys()
        self.mean = {col: X[col].mean() for col in self.cols}
        self.std = {col: X[col].std() for col in self.cols}

    def transform(self, X: pd.DataFrame, **kwargs):
        X = X.copy()

        for col in self.cols:
            X[col] = (X[col] - self.mean[col]) / self.std[col]

        return X


# Nouveautées


class TurnToNormDist:
    """
    La classe qui permet de transformer les variables pour qu'elles
    suivent une loie normale
    --------
    Ex: normal_distrib = TurnToNormDist(cols)
        normal_distrib.fit(X)
        normal_distrib.transform(X)

    """

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        pass

    def transform(self, X, **kwargs):

        data = X.copy()
        for col in self.cols:
            data[col] = power_transform(data[[col]])

        return data


def select_feat_by_corr(X, y, threshold=0.09):
    y.columns = ["Target"]
    data = pd.concat([X, y], axis=1)
    _corr = data.corr()[["Target"]].sort_values("Target")
    feat_to_drop = list(
        _corr[(_corr["Target"] < threshold) & (_corr["Target"] > -threshold)].index
    )
    # X.drop(feat_to_drop,axis=1,inplace=True)
    return feat_to_drop, _corr