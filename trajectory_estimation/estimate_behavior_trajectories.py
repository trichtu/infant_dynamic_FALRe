"""Estimate behavioral GAMMs in 1,000 infant-level 80% subsamples."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from GAMM_function import GAMM_trajectory_estimate


ROOT = Path(__file__).resolve().parent
DATA = '.'
OUTPUT = '.'


def load_behavior(filename, column):
    sessions = np.load(DATA / 'unpaired_session_list_5.npy').astype(str)
    gender = np.load(DATA / 'unpaired_demo_list_5.npy')[:, 0]
    demographics = pd.DataFrame({'subject': [s[:8] for s in sessions],
                                 'gender': gender})

    sex = demographics.drop_duplicates('subject').set_index('subject').gender
    frame = pd.read_excel(DATA / filename)
    frame['subject'] = frame.SubID.astype(str).str.replace('-', '_', regex=False).str[:8]
    grouped = frame.groupby(['subject', 'CorrAge'], as_index=False)[column].mean()
    grouped['gender'] = grouped.subject.map(sex)
    valid = (np.isfinite(grouped[column]) & np.isfinite(grouped.CorrAge)
             & grouped.gender.notna())
    excluded = grouped.loc[~valid].copy()
    return grouped.loc[valid].reset_index(drop=True), excluded


def estimate(label, filename, column):
    frame, excluded = load_behavior(filename, column)
    frame.to_csv(OUTPUT / f'{label}_input.csv', index=False)
    excluded.to_csv(OUTPUT / f'{label}_excluded.csv', index=False)
    subjects = np.sort(frame.subject.unique().astype(str))
    sample_size = int(np.floor(0.8 * len(subjects)))
    ages = np.linspace(frame.CorrAge.min(), frame.CorrAge.max(), 181)
    reference = dict(gender=str(frame.gender.astype(str).mode().iloc[0]),
                     motion=0.0, site='0.0')

    def fit(sample):
        fitted = GAMM_trajectory_estimate(
            sample[column].to_numpy(), sample.CorrAge.to_numpy(),
            sample.gender.to_numpy(), np.zeros(len(sample)), np.zeros(len(sample)),
            sample.subject.to_numpy(), subject_list=sample.subject.to_numpy(),
            gradients=True, x1_pred=ages, n_splines=4, reference=reference,
            max_outer=1000, fit_method='QR', check_cond=0)
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
    estimate('eyelooking', 'ETData_forLM_081825.xlsx', 'EyesFix')
    estimate('targets', 'saccAmp_targetsReached_Data_LMsamp_103125_5deg.xlsx',
             '% Targets Reached')
