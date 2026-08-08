import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def residualize(y, covariates):
    """
    Regress y on covariates and return residuals.
    """
    y = np.asarray(y, dtype=float).ravel()
    covariates = np.asarray(covariates, dtype=float)

    model = LinearRegression()
    model.fit(covariates, y)
    y_pred = model.predict(covariates)

    return y - y_pred


def fdr_bh(pvals):
    """
    Benjamini-Hochberg FDR correction.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)

    order = np.argsort(pvals)
    ranked = pvals[order]

    corrected = ranked * n / np.arange(1, n + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0, 1)

    p_fdr = np.empty_like(corrected)
    p_fdr[order] = corrected

    return p_fdr


def partial_corr(x, y, covariates):
    """
    Partial correlation between x and y, controlling for covariates.
    """
    x_res = residualize(x, covariates)
    y_res = residualize(y, covariates)

    r, _ = pearsonr(x_res, y_res)

    return r


def permutation_partial_corrs(
    X,
    y,
    feature_names=None,
    n_perm=10000,
    random_state=42,
    two_tailed=True
):
    """
    Permutation test for partial correlations between 5 features and one behavior.

    For each feature Xi:
        partial corr between Xi and y controlling for other features.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
    y : array, shape (n_samples,)
    feature_names : list of str
    n_perm : int
    random_state : int
    two_tailed : bool

    Returns
    -------
    results : DataFrame
        Feature, partial_r, permutation p, FDR-corrected p.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    n_samples, n_features = X.shape

    if feature_names is None:
        feature_names = [f"F{i+1}" for i in range(n_features)]

    rng = np.random.default_rng(random_state)

    observed_r = []
    p_perm = []

    for i in range(n_features):
        others = [j for j in range(n_features) if j != i]

        x_i = X[:, i]
        covariates = X[:, others]

        # observed partial correlation
        r_obs = partial_corr(x_i, y, covariates)
        observed_r.append(r_obs)

        # permutation null distribution
        null_rs = np.zeros(n_perm)

        for p in range(n_perm):
            y_perm = rng.permutation(y)
            null_rs[p] = partial_corr(x_i, y_perm, covariates)

        if two_tailed:
            # two-tailed test
            p_val = (np.sum(np.abs(null_rs) >= np.abs(r_obs)) + 1) / (n_perm + 1)
        else:
            # one-tailed test, positive direction
            p_val = (np.sum(null_rs >= r_obs) + 1) / (n_perm + 1)

        p_perm.append(p_val)

    p_fdr = fdr_bh(p_perm)

    results = pd.DataFrame({
        "Feature": feature_names,
        "Partial_r": observed_r,
        "p_perm": p_perm,
        "p_FDR": p_fdr,
        "Significant_FDR_0.05": p_fdr < 0.05
    })

    return results