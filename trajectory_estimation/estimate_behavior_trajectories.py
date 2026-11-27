"""Estimate behavioral GAMMs in 1,000 infant-level 80% subsamples."""
from pathlib import Path

import numpy as np
import pandas as pd

from GAMM_function import GAMM_trajectory_estimate


ROOT = Path(__file__).resolve().parent
DATA = ROOT
OUTPUT = ROOT


def load_behavior(filename, column):
    """Read saved infant-age averages with their age and sex covariates."""
    frame = pd.read_csv(DATA / filename, dtype={'subject': str})
    frame = frame[['subject', 'CorrAge', column, 'gender']]
    valid = (np.isfinite(frame[column]) & np.isfinite(frame.CorrAge)
             & frame.gender.notna() & frame.subject.notna())
    return frame.loc[valid].reset_index(drop=True)


def estimate(label, filename, column):
    frame= load_behavior(filename, column)
    subjects = np.sort(frame.subject.unique().astype(str))
    sample_size = int(np.floor(0.8 * len(subjects)))
    ages = np.linspace(frame.CorrAge.min(), frame.CorrAge.max(), 181)
    reference = dict(gender=str(frame.gender.astype(str).mode().iloc[0]),
                     motion=0.0, site='0.0')

    def fit(sample):
        fitted = GAMM_trajectory_estimate(
            sample[column].to_numpy(), sample.CorrAge.to_numpy(),
            sample.gender.to_numpy(), np.zeros(len(sample)), np.zeros(len(sample)),
            sample.subject.to_numpy(),
            gradients=True, x1_pred=ages, n_splines=4, reference=reference,
            max_outer=1000, check_cond=0)
        if fitted[0].info.code != 0:
            raise RuntimeError(f'GAMM did not converge: code={fitted[0].info.code}')
        condition = np.linalg.cond(fitted[0].lvi.toarray())
        if not np.isfinite(condition) or condition > 1 / np.sqrt(np.finfo(float).eps / 2):
            raise RuntimeError(f'Ill-conditioned inverse factor: {condition}')
        fitted[0].dense_condition_ = condition
        return fitted

    full = fit(frame)
    curves = np.full((1000, len(ages)), np.nan)
    derivatives = np.full_like(curves, np.nan)
    selections = np.empty((1000, sample_size), dtype=subjects.dtype)
    failures = []
    iterations = np.full(1000, -1, dtype=int)
    conditions = np.full(1000, np.nan)
    rng = np.random.default_rng(42)
    print(f'{label}: {len(frame)} visits, {len(subjects)} infants, '
          f'{sample_size} infants per subsample', flush=True)
    for index in range(1000):
        selections[index] = rng.choice(subjects, sample_size, replace=False)
        sample = frame.loc[frame.subject.isin(selections[index])]
        try:
            model, curve, _, derivative = fit(sample)
            iterations[index] = model.info.iter
            conditions[index] = model.dense_condition_
            inside = (ages >= sample.CorrAge.min()) & (ages <= sample.CorrAge.max())
            curves[index, inside] = curve[inside]
            derivatives[index, inside] = derivative[inside]
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            failures.append({'replicate': index, 'error': str(error)})
        if (index + 1) % 100 == 0:
            print(f'{label}: {index + 1}/1000, failures={len(failures)}', flush=True)
    np.savez_compressed(
        OUTPUT / f'{label}_trajectories.npz', ages=ages,
        trajectory_bootstrap=curves, derivative_bootstrap=derivatives,
        full_trajectory=full[1], full_derivative=full[3], full_mean_ci=full[2],
        sampled_subjects=selections,
        fitting_iterations=iterations, dense_condition=conditions,
        n_valid=np.isfinite(curves).sum(axis=0))



if __name__ == '__main__':
    OUTPUT.mkdir(exist_ok=True)
    estimate('eyelooking', 'eyelooking_input.csv', 'EyesFix')
    estimate('targets', 'targets_input.csv',
             '% Targets Reached')
