import numpy as np
import pandas as pd
from mssm.models import GAMM, Formula, lhs, i, f, l, ri, Gaussian


def GAMM_trajectory_estimate(
    target, age_list, gender_list, hd_list, site_list, subject_list,
    gradients=False, x1_pred=None, n_splines=10, visit_list=None,
    width=0.95, max_outer=200, age_penalty_multiplier=1.0,
):
    """Gaussian GAMM for clean, row-aligned inputs without missing values.

    Fits an age smooth, nuisance effects and infant random intercepts.
    Optional visit IDs must be globally unique and nested within infants.
    Returns model, population curve, pointwise mean CI, and optionally its
    numerical age derivative. No manual smoothing lambda or weights are used.
    max_outer controls the REML fitting iteration limit (default 200).
    age_penalty_multiplier > 1 fixes the age penalty at that multiple of
    its REML estimate, then refits other penalties. Default 1 keeps REML.
    """
    data = pd.DataFrame({
        "target": np.asarray(target).ravel(),
        "age": np.asarray(age_list).ravel(),
        "gender": np.asarray(gender_list).ravel(),
        "motion": np.asarray(hd_list).ravel(),
        "site": np.asarray(site_list).ravel(),
        "subject": np.asarray(subject_list).ravel(),
    })
    # mssm requires object dtype for categorical predictors.
    for column in ["gender", "site", "subject"]:
        data[column] = data[column].astype(str).astype(object)

    terms = [i(), f(["age"], nk=n_splines)]
    for column in ["gender", "motion", "site"]:
        if data[column].nunique() > 1:
            terms.append(l([column]))
    population_terms = list(range(len(terms)))
    terms.append(ri("subject"))

    if visit_list is not None:
        data["visit"] = np.asarray(visit_list).ravel().astype(str).astype(object)
        terms.append(ri("visit"))

    model = GAMM(Formula(lhs("target"), terms, data=data), Gaussian())
    model.fit(method="QR", n_cores=1, progress_bar=False, max_outer=max_outer)
    if age_penalty_multiplier != 1.0:
        age_penalty = model.overall_penalties[0]
        age_penalty.lam *= age_penalty_multiplier
        age_penalty.frozen = True
        model.fit(method="QR", n_cores=1, progress_bar=False,
                  max_outer=max_outer, restart=True)

    if x1_pred is None:
        x1_pred = np.linspace(data["age"].min(), data["age"].max(), 181)
    x1_pred = np.asarray(x1_pred).ravel()
    prediction_data = pd.DataFrame({
        "age": x1_pred,
        "gender": data["gender"].mode().iloc[0],
        "motion": data["motion"].mean(),
        "site": data["site"].mode().iloc[0],
        "subject": data["subject"].iloc[0],
    })
    for column in ["gender", "site", "subject"]:
        prediction_data[column] = prediction_data[column].astype(object)
    if visit_list is not None:
        prediction_data["visit"] = data["visit"].iloc[0]
        prediction_data["visit"] = prediction_data["visit"].astype(object)

    # Exclude infant/visit offsets from the population curve and its CI.
    y_pred, _, half_width = model.predict(
        use_terms=population_terms, n_dat=prediction_data,
        ci=True, alpha=1-width, whole_interval=False,
    )
    y_pred = np.asarray(y_pred).ravel()
    half_width = np.asarray(half_width).ravel()
    conf_int = np.column_stack((y_pred-half_width, y_pred+half_width))
    model.prediction_ages_ = x1_pred

    if gradients:
        y_derivative = np.gradient(y_pred, x1_pred)
        return model, y_pred, conf_int, y_derivative
    return model, y_pred, conf_int
