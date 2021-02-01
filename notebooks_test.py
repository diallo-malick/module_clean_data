# Module importation
from explore import *
from training import *
from preprocessing import *
from feature_engineering import *
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
# Data importation
X, y = pd.read_csv("ref_train_x.csv"), pd.read_csv("ref_train_y.csv", header=None)
X.head()
X.drop("raw_id", axis=1, inplace=True)
# Testing
# Show missing values information
missings_values_infos(X)

# Getting of colonnes names containes nan values
var_nb_nan = X.isna().sum()
cols_with_nan = list(var_nb_nan[var_nb_nan > 0].index)

# Append sector to cols_with_nan for normalisation
cols_with_nan.append("sector")

# cat_col = X_train_cat.columns

# Initialisation of somes transformations


tmp = X.sample(1000).iloc[1:5, 1:5]
import numpy as np

tmp.iloc[1:3, 1] = np.nan
tmp.iloc[2:, 2] = np.nan
tmp
liste_trans = [
    ImputeWithMean(["earnings_implied_obs"]),
    ImputeWithMedian(["delta_vol_1w", "delta_vol_1y"]),
    StaticTransform(
        ["delta_vol_1w", "delta_vol_1y"],
        polynomial_features(d=2, interaction_only=False),
    ),
    MinMaxNormalization(
        polynomial_feature_name(
            tmp, cols=["delta_vol_1w", "delta_vol_1y"], interaction_only=False
        )
    ),
    One_Hot_Encoder(["sector"]),
]
tmp

compose_trans = ComposeTransforms(liste_trans)

compose_trans.fit(tmp)
compose_trans.transform(tmp)

tmp["earnings_implied_obs"].mean()


# Evaluation sur l'ensemble de nos donées

l_trans = [
    ImputeWithMean(cols_with_nan),
    # ImputeWithMedian(["delta_vol_1w", "delta_vol_1y"]),
    StaticTransform(
        ["delta_vol_1w", "delta_vol_1y"],
        polynomial_features(d=2, interaction_only=False),
    ),
    MinMaxNormalization(
        polynomial_feature_name(
            X, cols=["delta_vol_1w", "delta_vol_1y"], interaction_only=False
        )
    ),
    StdNormalization(cols_with_nan),
    One_Hot_Encoder(["exchange"]),
]

tr = ComposeTransforms(l_trans)

from sklearn.linear_model import LogisticRegression

# from sklearn.svm import SVC

from sklearn.ensemble import BaggingClassifier

clf_log = LogisticRegression()
clf_bag = BaggingClassifier(n_estimators=10)


models = [clf_bag, clf_log]

get_split_loader = get_split_loader_func(3, X)

evaluate(models, [tr], X, y, get_split_loader)


X.head


X.shape, y.shape