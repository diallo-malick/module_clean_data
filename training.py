# training.py
import os
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

# skf = StratifiedKFold(n_splits=5)
# skf.split(data, y)
# kf = KFold(n_splits=5)
# kf.split(data, y)

""""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier


lreg =  LogisticRegression(random_state=0)
bgg = BaggingClassifier(n_estimators=10, random_state=0)

split_loader = get_split_loader_func(n_splits=3, X=data_X)
evaluate([lreg, bgg], [tr], X, data_y, split_loader)
"""


def mkdir_if_not_exist(path):  # @save
    if not isinstance(path, str):
        path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)


def get_stratified_split(n_splits, X, y, shuffle=False):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    split_loader = skf.split(X, y)
    return split_loader


def get_split(n_splits, X, y=None, shuffle=False):
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    split_loader = kf.split(X)
    return split_loader


def get_split_loader_func(n_splits, X, y=None, shuffle=False):
    """
    get_split_loader = get_split_loader_fun(5, data)
    split_loader = get_split_loader()

    for train_index, val_index in get_split_loader:
        ...
    """

    if y is None:
        return lambda: get_split(n_splits, X, shuffle=shuffle)
    # else
    return lambda: get_stratified_split(n_splits, X, y, shuffle=shuffle)


def evaluate(models, transforms, data_X, data_y, get_split_loader, log_dir="Results"):
    """
    get_split_loader = get_split_loader_func(5, data_X)
    """
    t = int(time.time())
    log_file = f"{log_dir}/results_{t}.csv"

    mkdir_if_not_exist(log_dir)

    scores_df = []
    for model in models:
        for transform in transforms:
            start_time = time.time()

            split_loader = get_split_loader()
            scores, scores_tmp = cross_valide(
                model, data_X, data_y, transform, split_loader
            )
            # scores = cross_valide(model, data_X, data_y, transform, split_loader)
            scores["pipeline"] = {"model": model, "transform": transform}

            scores["time (min.)"] = int(time.time() - start_time) // 60

            scores_df.append(scores)

            pd.DataFrame(scores_df).to_csv(log_file, index=False)

    return pd.DataFrame(scores_df)


def cross_valide(model, dataX, datay, transform, split_loader):

    data_X = dataX.copy()
    data_y = datay.copy()

    scores = {
        "train mean acc": None,
        "train std acc": None,
        "val mean acc": None,
        "val std acc": None,
        "train mean roc_auc_score": None,
        "train std roc_auc_score": None,
        "val mean roc_auc_score": None,
        "val std roc_auc_score": None,
    }

    scores_tmp = {
        "train acc": [],
        "val acc": [],
        "train roc_auc_score": [],
        "val roc_auc_score": [],
    }

    # other metrics

    for train_index, val_index in split_loader:
        X_train = data_X.iloc[
            train_index, :
        ]  # pd.DataFrame(X_train, columns=data.columns)
        y_train = data_y.iloc[train_index, :].values.reshape(-1, 1).flatten()
        X_val = data_X.iloc[val_index, :]  # pd.DataFrame(X_val, columns=data.columns)
        y_val = data_y.iloc[val_index, :].values.reshape(-1, 1).flatten()

        ### Fitting

        transform.fit(X_train, y_train)
        X_train = transform.transform(X_train, is_training=True)
        X_val = transform.transform(X_val)

        model.fit(X_train, y_train)
        # Prediction

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        y_train_pred_prob = model.predict_proba(X_train)[:, 1]
        y_val_pred_prob = model.predict_proba(X_val)[:, 1]

        ### Performances

        # =============== Acc
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_val = accuracy_score(y_val, y_val_pred)

        scores_tmp["train acc"].append(acc_train)
        scores_tmp["val acc"].append(acc_val)

        # =============== Roc
        roc_train = roc_auc_score(y_train, y_train_pred_prob)
        roc_val = roc_auc_score(y_val, y_val_pred_prob)

        scores_tmp["train roc_auc_score"].append(roc_train)
        scores_tmp["val roc_auc_score"].append(roc_val)

    # =============== Acc
    scores["train mean acc"] = np.mean(scores_tmp["train acc"])
    scores["val mean acc"] = np.mean(scores_tmp["val acc"])
    scores["train std acc"] = np.std(scores_tmp["train acc"])
    scores["val std acc"] = np.std(scores_tmp["val acc"])

    # =============== ROC
    scores["train mean roc_auc_score"] = np.mean(scores_tmp["train roc_auc_score"])
    scores["val mean roc_auc_score"] = np.mean(scores_tmp["val roc_auc_score"])
    scores["train std roc_auc_score"] = np.std(scores_tmp["train roc_auc_score"])
    scores["val std roc_auc_score"] = np.std(scores_tmp["val roc_auc_score"])

    return scores, scores_tmp
