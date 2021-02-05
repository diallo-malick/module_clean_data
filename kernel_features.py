import pandas as pd

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array, check_random_state, as_float_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS


class KernelFeature(TransformerMixin, BaseEstimator):
    def __init__(self, cols=None, prefix="kernel", keep_cols=False, kernel="rbf", *, gamma=None, coef0=None, degree=None,
                 kernel_params=None, n_components=100, random_state=None,
                 n_jobs=None):

        self.cols = cols
        self.keep_cols = keep_cols
        self.prefix = prefix

        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs
    
    def fit(self, X, y=None):
        """Fit estimator to data.
        Samples a subset of training points, computes kernel
        on these and computes normalization matrix.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        if self.cols is None:
            self.cols = X.columns
        
        #X = self._validate_data(X, accept_sparse='csr')
        #rnd = check_random_state(self.random_state)

        self.basis = X[self.cols].copy()
    
    def transform(self, X, **kwargs):
        X = X.copy()

        gram_matrix = pairwise_kernels(X[self.cols], self.basis,
                                        metric=self.kernel,
                                        filter_params=True,
                                        n_jobs=self.n_jobs,
                                        **self._get_kernel_params()
                                    )
        
        kernel_cols = [f"{self.prefix}{i}" for i in range(gram_matrix.shape[1])]

        kernel_feature = pd.DataFrame(gram_matrix, columns=kernel_cols)

        if self.keep_cols:
            cols_to_keep = list(X.columns) + kernel_cols
        else:
            cols_to_keep = [col for col in X.columns if col not in self.cols] + kernel_cols
            
        X_tr = pd.concat([ X.reset_index(drop=True), kernel_feature ], axis=1)
        return X_tr[cols_to_keep]


    def _get_kernel_params(self):
        params = self.kernel_params
        if params is None:
            params = {}
        if not callable(self.kernel) and self.kernel != 'precomputed':
            for param in (KERNEL_PARAMS[self.kernel]):
                if getattr(self, param) is not None:
                    params[param] = getattr(self, param)
        else:
            if (self.gamma is not None or
                    self.coef0 is not None or
                    self.degree is not None):
                raise ValueError("Don't pass gamma, coef0 or degree to "
                                 "Nystroem if using a callable "
                                 "or precomputed kernel")

        return params
