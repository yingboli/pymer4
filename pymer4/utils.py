from __future__ import division

__all__ = [
    "get_resource_path",
    "_check_random_state",
    "_sig_stars",
    "_robust_estimator",
    "_chunk_boot_ols_coefs",
    "_chunk_perm_ols",
    "_permute_sign",
    "_ols",
    "_ols_group",
    "_corr_group",
    "_perm_find",
    "_to_ranks_by_group",
    "isPSD",
    "nearestPSD",
    "upper",
    "R2con",
    "con2R"
]

__author__ = ["Eshin Jolly"]
__license__ = "MIT"

import os 
import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy.stats import chi2
from rpy2.robjects.packages import importr

base = importr("base")
MAX_INT = np.iinfo(np.int32).max


def get_resource_path():
    """ Get path sample data directory. """
    return os.path.join(os.path.dirname(__file__), "resources") + os.path.sep


def _mean_diff(x, y):
    """For use in plotting of tost_equivalence"""
    return np.mean(x) - np.mean(y)


def _check_random_state(seed):
    """Turn seed into a np.random.RandomState instance. Note: credit for this code goes entirely to sklearn.utils.check_random_state. Using the source here simply avoids an unecessary dependency.

    Args:
        seed (None, int, np.RandomState): iff seed is None, return the RandomState singleton used by np.random. If seed is an int, return a new RandomState instance seeded with seed. If seed is already a RandomState instance, return it. Otherwise raise ValueError.
    """

    import numbers

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
    )


def _sig_stars(val):
    """Adds sig stars to coef table prettier output."""
    star = ""
    if 0 <= val < 0.001:
        star = "***"
    elif 0.001 <= val < 0.01:
        star = "**"
    elif 0.01 <= val < 0.05:
        star = "*"
    elif 0.05 <= val < 0.1:
        star = "."
    return star


def _robust_estimator(vals, X, robust_estimator="hc1", n_lags=1, cluster=None):
    """
    Computes robust sandwich estimators for standard errors used in OLS computation. Types include:
    'hc0': Huber (1980) sandwich estimator to return robust standard error estimates.
    'hc1': small sample dof correction to 'hc0'
    'hc2': alternate small sample weighting correction to 'hc0'
    'hc3': MacKinnon and White (1985) HC3 sandwich estimator. Provides more robustness in smaller samples than HC0 and HC1 Long & Ervin (2000)
    'hac': Newey-West (1987) estimator for robustness to heteroscedasticity as well as serial auto-correlation at given lags.
    Good reference: https://bit.ly/2VRb7jK

    Args:
        vals (np.ndarray): 1d array of residuals
        X (np.ndarray): design matrix used in OLS
        robust_estimator (str): estimator type, 'hc0' (default), 'hc3', 'hac', or 'cluster'
        n_lags (int): number of lags, only used with 'hac' estimator, default is 1
        cluster (np.ndarry): array of cluster ids

    Returns:
        stderr (np.ndarray): 1d array of standard errors with length == X.shape[1]

    """

    assert robust_estimator in [
        "hc0",
        "hc1",
        "hc2",
        "hc3",
        "hac",
        "cluster",
    ], "robust_estimator must be one of hc0, hc1, hc2, hc3, hac, or cluster"

    # Make a sandwich!
    # First we need bread
    bread = np.linalg.pinv(np.dot(X.T, X))

    # Then we need meat
    # First deal with estimators that have more complicated formulations

    # Cluster robust
    if robust_estimator == "cluster":
        # Good ref: http://projects.iq.harvard.edu/files/gov2001/files/sesection_5.pdf
        if cluster is None:
            raise ValueError("data column identifying clusters must be provided")
        else:
            u = vals[:, np.newaxis] * X
            u = pd.DataFrame(u)
            # Use pandas groupby to get cluster-specific residuals
            u["Group"] = cluster
            u_clust = u.groupby("Group").sum()
            num_grps = u["Group"].nunique()
            meat = (
                (num_grps / (num_grps - 1))
                * (X.shape[0] / (X.shape[0] - X.shape[1]))
                * u_clust.T.dot(u_clust)
            )

    # Auto-correlation robust
    elif robust_estimator == "hac":
        weights = 1 - np.arange(n_lags + 1.0) / (n_lags + 1.0)

        # First compute lag 0
        V = np.diag(vals ** 2)
        meat = weights[0] * np.dot(np.dot(X.T, V), X)

        # Now loop over additional lags
        for l in range(1, n_lags + 1):

            V = np.diag(vals[l:] * vals[:-l])
            meat_1 = np.dot(np.dot(X[l:].T, V), X[:-l])
            meat_2 = np.dot(np.dot(X[:-l].T, V), X[l:])

            meat += weights[l] * (meat_1 + meat_2)

    else:
        # Otherwise deal with estimators that modify the same essential operation
        V = np.diag(vals ** 2)

        if robust_estimator == "hc0":
            # No modification of residuals
            pass

        elif robust_estimator == "hc1":
            # Degrees of freedom adjustment to HC0
            V = V * X.shape[0] / (X.shape[0] - X.shape[1])

        elif robust_estimator == "hc2":
            # Rather than dof correction, weight residuals by reciprocal of "leverage values" in the hat-matrix
            V = V / (1 - np.diag(np.dot(X, np.dot(bread, X.T))))

        elif robust_estimator == "hc3":
            # Same as hc2 but more aggressive weighting due to squaring
            V = V / (1 - np.diag(np.dot(X, np.dot(bread, X.T)))) ** 2

        meat = np.dot(np.dot(X.T, V), X)
    # Finally we make a sandwich
    vcv = np.dot(np.dot(bread, meat), bread)

    return np.sqrt(np.diag(vcv))


def _whiten_wls(mat, weights):
    """
    Whiten a matrix for a WLS regression. Just multiply each column of mat by sqrt(weights) if mat is 2d. Similar to statsmodels

    Args:
        x (np.ndarray): design matrix to be passed to _ols
        weights (np.ndarray): 1d array of weights, most often variance of each group if some columns in x refer to categorical predictors
    """

    if weights.shape[0] != mat.shape[0]:
        raise ValueError(
            "The number of weights must be the same as the number of observations"
        )
    if mat.ndim == 1:
        return mat * np.sqrt(weights)
    elif mat.ndim == 2:
        # return np.column_stack([x[:,0], np.sqrt(weights)[:, None]*x[:,1:]])
        return np.sqrt(weights)[:, None] * mat


def _ols(x, y, robust, n_lags, cluster, all_stats=True, resid_only=False, weights=None):
    """
    Compute OLS on data. Useful for single computation and within permutation schemes.
    """

    if all_stats and resid_only:
        raise ValueError("_ols must be called with EITHER all_stats OR resid_only")
    # Expects as input pandas series and dataframe
    Y, X = y.values.squeeze(), x.values

    # Whiten if required
    if weights is not None:
        if isinstance(weights, (pd.DataFrame, pd.Series)):
            weights = weights.values
        X = _whiten_wls(X, weights)
        Y = _whiten_wls(Y, weights)

    # The good stuff
    b = np.dot(np.linalg.pinv(X), Y)

    if all_stats:

        res = Y - np.dot(X, b)

        if robust:
            se = _robust_estimator(
                res, X, robust_estimator=robust, n_lags=n_lags, cluster=cluster
            )
        else:
            sigma = np.sqrt(res.T.dot(res) / (X.shape[0] - X.shape[1]))
            se = np.sqrt(np.diag(np.linalg.pinv(np.dot(X.T, X)))) * sigma

        t = b / se

        return b, se, t, res

    elif resid_only:
        return Y - np.dot(X, b)
    else:
        return b


def _chunk_perm_ols(x, y, robust, n_lags, cluster, weights, seed):
    """
    Permuted OLS chunk.
    """
    # Shuffle y labels
    y = y.sample(frac=1, replace=False, random_state=seed)
    b, s, t, res = _ols(x, y, robust, n_lags, cluster, weights=weights, all_stats=True)

    return list(t)


def _permute_sign(data, seed, return_stat="mean"):
    """Given a list/array of data, randomly sign flip the values and compute a new mean. For use in one-sample permutation test. Returns a 'mean' or 't-stat'."""

    random_state = np.random.RandomState(seed)
    new_dat = data * random_state.choice([1, -1], len(data))
    if return_stat == "ceof":
        return np.mean(new_dat)
    elif return_stat == "t-stat":
        return np.mean(new_dat) / (np.std(new_dat, ddof=1) / np.sqrt(len(new_dat)))


def _chunk_boot_ols_coefs(dat, formula, weights, seed):
    """
    OLS computation of coefficients to be used in a parallelization context.
    """
    # Random sample with replacement from all data
    dat = dat.sample(frac=1, replace=True, random_state=seed)
    y, x = dmatrices(formula, dat, 1, return_type="dataframe")
    b = _ols(
        x, y, robust=None, n_lags=1, cluster=None, all_stats=False, weights=weights
    )
    return list(b)


def _ols_group(dat, formula, group_col, group, rank):
    """Compute OLS on data given a formula. Used by Lm2"""
    dat = dat[dat[group_col] == group].reset_index(drop=True)
    if rank:
        dat = dat.rank()
    y, x = dmatrices(formula, dat, 1, return_type="dataframe")
    b = _ols(x, y, robust=None, n_lags=1, cluster=None, all_stats=False)
    return list(b)


def _corr_group(dat, formula, group_col, group, rank, corr_type):
    """Compute partial correlations via OLS. Used by Lm2"""

    from scipy.stats import pearsonr

    dat = dat[dat[group_col] == group].reset_index(drop=True)
    if rank:
        dat = dat.rank()
    y, x = dmatrices(formula, dat, 1, return_type="dataframe")
    corrs = []
    for c in x.columns[1:]:
        other_preds = [e for e in x.columns if e != c]
        other_preds = x[other_preds]
        cc = x[c]
        pred_m_resid = _ols(
            other_preds,
            cc,
            robust=None,
            n_lags=1,
            cluster=None,
            all_stats=False,
            resid_only=True,
        )
        if corr_type == "semi":
            dv_m_resid = y.values.squeeze()
        elif corr_type == "partial":
            dv_m_resid = _ols(
                other_preds,
                y,
                robust=None,
                n_lags=1,
                cluster=None,
                all_stats=False,
                resid_only=True,
            )
        corrs.append(pearsonr(dv_m_resid, pred_m_resid)[0])
    return corrs


def _to_ranks_by_group(dat, group, formula, exclude_cols=[]):
    """
    Covert predictors to ranks separately for each group for use in rank Lmer. Any columns not in the model formula or in exclude_cols will not be converted to ranks. Used by models.Lmer

    Args:
        dat (pd.DataFrame): dataframe of data
        group (string): string name of column to group data on
        formula (string): Lmer flavored model formula with random effects
        exclude_cols (list): optional columns that are part of the formula to exclude from rank conversion.

    Returns:
        pandas.core.frame.DataFrame: ranked data

    """

    if (not isinstance(group, str)) and (group not in dat.columns):
        raise TypeError(
            "group must be a valid column name in the dataframe. Currently only 1 grouping variable is supported."
        )
    if isinstance(exclude_cols, str):
        exclude_cols = [exclude_cols]
    original_col_order = list(dat.columns)
    formula = formula.replace(" ", "")
    to_rank = formula.split("~")[-1].split("(")[0].split("+")[:-1]
    # add dv to be ranked
    to_rank.append(formula.split("~")[0])
    to_rank = [c for c in to_rank if c not in exclude_cols]
    other_cols = [c for c in dat.columns if c not in to_rank]
    dat = pd.concat(
        [dat[other_cols], dat.groupby(group).apply(lambda g: g[to_rank].rank())], axis=1
    )
    return dat[original_col_order]


def _perm_find(arr, x):
    """
    Find permutation cutoff in array. Two-tailed only
    """
    return (np.sum(np.abs(arr) >= np.abs(x)) + 1) / (float(len(arr)) + 1)


def isPSD(mat, tol=1e-8):
    """
    Check if matrix is positive-semi-definite by virtue of all its eigenvalues being >= 0. The cholesky decomposition does not work for edge cases because np.linalg.cholesky fails on matrices with exactly 0 valued eigenvalues, whereas in Matlab this is not true, so that method appropriate. Ref: https://goo.gl/qKWWzJ

    Args:
        mat (np.ndarray): 2d numpy array

    Returns:
        bool: whether matrix is postive-semi-definite
    """

    # We dont assume matrix is Hermitian, i.e. real-valued and symmetric
    # Could swap this out with np.linalg.eigvalsh(), which is faster but less general
    e = np.linalg.eigvals(mat)
    return np.all(e > -tol)


def nearestPSD(mat, nit=100):
    """
    Higham (2000) algorithm to find the nearest positive semi-definite matrix that minimizes the Frobenius distance/norm. Statsmodels using something very similar in corr_nearest(), but with spectral SGD to search for a local minima. Reference: https://goo.gl/Eut7UU

    Args:
        mat (np.ndarray): 2d numpy array
        nit (int): number of iterations to run algorithm; more iterations improves accuracy but increases computation time.
    Returns:
        np.ndarray: closest positive-semi-definite 2d numpy array
    """

    n = mat.shape[0]
    W = np.identity(n)

    def _getAplus(mat):
        eigval, eigvec = np.linalg.eig(mat)
        Q = np.matrix(eigvec)
        xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
        return Q * xdiag * Q.T

    def _getPs(mat, W=None):
        W05 = np.matrix(W ** 0.5)
        return W05.I * _getAplus(W05 * mat * W05) * W05.I

    def _getPu(mat, W=None):
        Aret = np.array(mat.copy())
        Aret[W > 0] = np.array(W)[W > 0]
        return np.matrix(Aret)

    # W is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    deltaS = 0
    Yk = mat.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    # Double check returned matrix is PSD
    if isPSD(Yk):
        return Yk
    else:
        nearestPSD(Yk)


def upper(mat):
    """
    Return upper triangle of matrix. Useful for grabbing unique values from a symmetric matrix.

    Args:
        mat (np.ndarray): 2d numpy array
    
    Returns:
        np.array: 1d numpy array of values
    
    """
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]


def _return_t(model):
    """Return t or z stat from R model summary."""
    summary = base.summary(model)
    unsum = base.unclass(summary)
    return unsum.rx2("coefficients")[:, -1]


def _get_params(model):
    """Get number of params in a model."""
    return model.coefs.shape[0]


def _lrt(tup):
    """Likelihood ratio test between 2 models. Used by stats.lrt"""
    d = np.abs(2 * (tup[0].logLike - tup[1].logLike))
    return chi2.sf(d, np.abs(tup[0].coefs.shape[0] - tup[1].coefs.shape[0]))


def _welch_ingredients(x):
    """
    Helper function to compute the numerator and denominator for a single group/array for use in Welch's degrees of freedom calculation. Used by stats.welch_dof
    """

    numerator = x.var(ddof=1) / x.size
    denominator = np.power(x.var(ddof=1) / x.size, 2) / (x.size - 1)
    return [numerator, denominator]


def con2R(arr):
    """
    Convert user desired contrasts to R-flavored contrast matrix that can be passed directly to lm(). Reference: https://goo.gl/E4Mms2

    Args:
        arr (np.ndarry): 2d numpy array arranged as contrasts X factor levels

    Returns:
        np.ndarray: 2d contrast matrix as expected by R's contrasts() function
    """

    intercept = np.repeat(1.0 / arr.shape[1], arr.shape[1])
    mat = np.vstack([intercept, arr])
    inv = np.linalg.inv(mat)[:, 1:]
    return inv


def R2con(arr):
    """
    Convert R-flavored contrast matrix to intepretable contrasts as would be specified by user. Reference: https://goo.gl/E4Mms2

    Args:
        arr (np.ndarry): 2d contrast matrix output from R's contrasts() function.

    Returns:
        np.ndarray: 2d array organized as contrasts X factor levels
    """

    intercept = np.ones((arr.shape[0], 1))
    mat = np.column_stack([intercept, arr])
    inv = np.linalg.inv(mat)
    return inv


def _df_meta_to_arr(df):
    """Check what kind of data exists in pandas columns or index. If string return as numpy array 'S' type, otherwise regular numpy array.
    """
    
    if len(df.columns):
        if isinstance(df.columns[0], str):
            columns = df.columns.values.astype("S")
        else:
            columns = df.columns.values
    else:
        columns = []
    
    if len(df.index):
        if isinstance(df.index[0], str):
            index = df.index.values.astype("S")
        else:
            index = df.index.values
    else:
        index = []

    return columns, index
