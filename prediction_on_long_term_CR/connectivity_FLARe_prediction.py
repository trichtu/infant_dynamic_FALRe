"""FLARe-only CR prediction from row-aligned MATLAB exports."""
import argparse
import csv
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import false_discovery_control, t

DIRECTIONS = ('future', 'previous')
CONNECTIVITY = {'connectivity': 'connectivity'}
BEHAVIOR = {'targets': 'behavior_targets', 'eyelooking': 'behavior_eyelooking'}


def prepare(folder, datasets, cache):
    arrays = {}
    for name, stem in datasets.items():
        for direction in DIRECTIONS:
            path = folder / f'{direction}_{stem}.mat'
            mat = loadmat(path)
            flare = mat['FLARe']
            target = mat[f'{direction}_CR_rate']
            subjects = np.char.strip(mat['subjects' if stem == 'connectivity' else 'subID'].astype(str).ravel())
            if target.shape != (len(subjects), 28 if stem == 'connectivity' else 1) or flare.shape != (len(subjects), 10 if stem == 'connectivity' else 5):
                raise ValueError(f'Unexpected row or column counts in {path}')
            raw = flare[:, :5]
            flare = np.column_stack([raw, raw**2])
            if not np.isfinite(flare).all() or not np.isfinite(target).all() or np.any(subjects == ''):
                raise ValueError(f'Invalid values in {path}')
            key = f'{name}_{direction}'
            arrays.update({f'{key}_X': flare, f'{key}_y': target, f'{key}_subjects': subjects})
            print(f'{key}: {len(subjects)} rows, {len(np.unique(subjects))} subjects')
    temporary = cache.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(cache)


def predict(cache, output, repeat, folds, trees):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold

    results = {'folds': np.array(folds), 'cache_hash': np.array(sha256(cache.read_bytes()).hexdigest())}
    with np.load(cache, allow_pickle=False) as data:
        for field in data.files:
            if not field.endswith('_X'):
                continue
            key = field[:-2]
            X, y = data[field], data[f'{key}_y']
            subjects = data[f'{key}_subjects']
            unique, groups = np.unique(subjects, return_inverse=True)
            cv = KFold(folds, shuffle=True, random_state=42 + repeat)
            fitted = np.empty_like(y, dtype=float)
            for train, test in cv.split(unique):
                train_rows = np.isin(groups, train)
                test_rows = np.isin(groups, test)
                for column in range(y.shape[1]):
                    model = RandomForestRegressor(n_estimators=trees, random_state=42)
                    model.fit(X[train_rows], y[train_rows, column])
                    fitted[test_rows, column] = model.predict(X[test_rows])
            results[key] = fitted
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'repeat_{repeat:04d}.npz'
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **results)
    temporary.replace(path)


def aggregate(cache, output, datasets):
    files = sorted(output.glob('repeat_*.npz'))
    if len(files) < 2:
        raise ValueError('At least two completed repeats are required.')
    cache_hash = sha256(cache.read_bytes()).hexdigest()
    repeats = []
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            if str(data['cache_hash']) != cache_hash:
                raise ValueError(f'{file}: predictions were generated from a different input cache.')
            repeats.append(dict(data))
    folds, = {int(data['folds']) for data in repeats}
    rows = []
    with np.load(cache, allow_pickle=False) as data:
        for direction in DIRECTIONS:
            family = []
            for name in datasets:
                key = f'{name}_{direction}'
                y = data[f'{key}_y']
                subjects = data[f'{key}_subjects']
                predictions = np.stack([entry[key] for entry in repeats])
                if predictions.shape[1:] != y.shape:
                    raise ValueError(f'{key}: repeat predictions do not match cached targets.')
                for column, observed in enumerate(y.T):
                    pred = predictions[:, :, column]
                    r = np.array([np.corrcoef(values, observed)[0, 1] for values in pred])
                    if not np.isfinite(r).all():
                        raise ValueError(f'{key}, column {column}: undefined correlation.')
                    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
                    # Same repeated-CV correction and one-sided test used in the comparison scripts.
                    se = z.std(ddof=1) * np.sqrt(1 / len(z) + 1 / (folds - 1))
                    center = z.mean()
                    p = t.sf(center / se, len(z)-1) if se else float(center <= 0)
                    lo, hi = np.tanh(center + np.array([-1, 1]) * t.ppf(.95, len(z)-1) * se)
                    mae = np.abs(pred - observed).mean()
                    zero = np.abs(observed).mean()
                    family.append(dict(direction=direction, outcome=name,
                                       target_column=column, samples=len(y),
                                       subjects=len(np.unique(subjects)), repeats=len(z),
                                       mean_r=f'{r.mean():.3f}', CI=f'[{lo:.3f}, {hi:.3f}]',
                                       MAE=f'{mae:.3f}', RMAE=f'{mae / zero:.3f}' if zero else '',
                                       p=p))
            adjusted = false_discovery_control([row['p'] for row in family])
            for row, p_fdr in zip(family, adjusted):
                row['p_FDR'] = '<0.001' if p_fdr < .001 else f'{p_fdr:.3f}'
            rows.extend(family)
    path = output / 'significance.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} rows to {path}')


def main(datasets=CONNECTIVITY, kind='connectivity'):
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=folder)
    parser.add_argument('--output', type=Path, default=folder / f'FLARe_{kind}_prediction_sample.npz')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--aggregate', action='store_true')
    parser.add_argument('--repeat-id', type=int, default=0)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--trees', type=int, default=50)
    args = parser.parse_args()
    cache = args.folder / f'FLARe_only_{kind}.npz'
    if args.prepare:
        prepare(args.folder, datasets, cache)
    elif args.aggregate:
        aggregate(cache, args.output, datasets)
    else:
        predict(cache, args.output, args.repeat_id, args.folds, args.trees)


if __name__ == '__main__':
    main()
