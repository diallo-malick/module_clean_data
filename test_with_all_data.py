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
X, y = pd.read_csv("data/ref_train_x.csv"), pd.read_csv("data/ref_train_y.csv", header=None)
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

from sklearn.ensemble import RandomForestClassifier

rd_clf = RandomForestClassifier(n_estimators=10)

new_trans = [
    ImputeWithMean(cols_with_nan),
    TreeFeature(cols=cols_with_nan, tree_model=rd_clf),
    One_Hot_Encoder(["exchange"])
]

new_tr = ComposeTransforms(new_trans)
new_tr.fit(X=X, y=y.values)
a = new_tr.transform(X)
a.head()
tr = ComposeTransforms(l_trans)
tr.fit(X)
tr.transform(X)
# Evaluation de rnd forest
get_split_loader = get_split_loader_func(3, X)
evaluate([clf_log], [new_tr], X, y, get_split_loader)
y.shape, X.shape
from sklearn.linear_model import LogisticRegression
type(y)
# from sklearn.svm import SVC

from sklearn.ensemble import BaggingClassifier

clf_log = LogisticRegression()
clf_bag = BaggingClassifier(n_estimators=10)


models = [clf_bag, clf_log]

get_split_loader = get_split_loader_func(3, X)

#evaluate(models, [tr], X, y, get_split_loader)

from sklearn.preprocessing import power_transform

    
nor_dis = TurnToNormDist(cols_with_nan)

new_X = nor_dis.transform(X)
new_X.shape==X.shape
power_transform(X[["sector"]])
X.head()
import seaborn as sns
sns.distplot( new_X["return_1w"])

def select_feat_by_corr(X, y, threshold=0.09):
    y.columns = ["Target"]
    data = pd.concat([X,y], axis=1)
    _corr = data.corr()[["Target"]].sort_values("Target")
    feat_to_drop=list(_corr[(_corr["Target"]< threshold)& (_corr["Target"]>-threshold)].index)
    #X.drop(feat_to_drop,axis=1,inplace=True)
    return feat_to_drop, _corr

corr_cols, a = select_feat_by_corr(X, y, 0.2)
a
pd.concat([X,y], axis=1)
y[0]
y.columns = ["Target"]
len(corr_cols)