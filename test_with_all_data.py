%load_ext autoreload
#for reload module after modif
%autoreload

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


var_nb_nan = X.isna().sum()
cols_with_nan = list(var_nb_nan[var_nb_nan > 0].index)

# Append sector to cols_with_nan for normalisation
cols_with_nan.append("sector")

l_trans = [
    TurnToNormDist(cols_with_nan),

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

from sklearn.preprocessing import power_transform

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
    
nor_dis = TurnToNormDist(cols_with_nan)

new_X = nor_dis.transform(X)
new_X.shape==X.shape
power_transform(X[["sector"]])
X.head()
import seaborn as sns
sns.distplot( new_X["return_1w"])