import numpy as np
from pathlib import Path
from scipy.io import loadmat, savemat
import subprocess


def dynamic_corr(timeseries_list, TR_list, slide_window=45):
    print('Calculate dynamic FNC in ', len(timeseries_list), ' sessions')
    whole_dFNC = []
    for timeseries, TR in zip(timeseries_list, TR_list):
        slice_range = int(np.ceil(slide_window/TR))
        dynamic_matrix = []
        for i in range(len(timeseries)-slice_range+1): 
            window_data = timeseries[i:i+slice_range,:]
            corr = np.corrcoef(window_data.T)
            # print(window_data.shape, corr.shape)
            dynamic_matrix.append(corr)
        dynamic_matrix = np.array(dynamic_matrix)
        whole_dFNC.append(dynamic_matrix)

    return whole_dFNC 


def decompose_dFNC(network_timeseries_list, TR_list, savedir, string, n_networks=53, sliding_window=45):
    if len(network_timeseries_list) != len(TR_list) or len(TR_list) == 0:
        raise ValueError('Provide one TR per session and at least one session.')

    sessions = []
    for source, tr in zip(network_timeseries_list, TR_list):
        if isinstance(source, (str, Path)):
            timeseries = loadmat(source)['timeseries']
        else:
            timeseries = np.asarray(source)
        if timeseries.ndim != 2:
            raise ValueError('Each session must be a two-dimensional array.')
        if timeseries.shape[1] != n_networks and timeseries.shape[0] == n_networks:
            timeseries = timeseries.T
        if timeseries.shape[1] != n_networks:
            raise ValueError(f'Each session must contain {n_networks} networks.')
        if not np.isfinite(tr) or tr <= 0 or not 2 <= np.ceil(sliding_window / tr) <= len(timeseries):
            raise ValueError('Each TR must be positive and the window must contain at least two scans and fit in the session.')
        sessions.append(timeseries)

    matrices = dynamic_corr(sessions, TR_list, slide_window=sliding_window)
    dFNC_list = []
    rows, cols = np.triu_indices(n_networks, k=1)
    for dFNC in matrices:
        edges = dFNC[:, rows, cols].T
        demeaned_edges = edges - edges.mean(axis=1, keepdims=True)
        dFNC_list.append(demeaned_edges)

    dFNC_list = np.concatenate(dFNC_list, axis=1)

    savedir = Path(savedir).resolve()
    savedir.mkdir(parents=True, exist_ok=True)
    total_dFNC_path = savedir / 'dFNC_total.mat'
    savemat(total_dFNC_path, {'timeseries': dFNC_list})

    ncomps = 10
    trials = 100
    matlab_dir = Path(__file__).resolve().parent
    arguments = (matlab_dir, total_dFNC_path, savedir, string)
    quoted = ["'" + str(value).replace("'", "''") + "'" for value in arguments]
    command = f'addpath({quoted[0]}); run_dfnc_icasso({quoted[1]}, {ncomps}, {trials}, {quoted[2]}, {quoted[3]});'
    subprocess.run(['matlab', '-batch', command], check=True)

    result = loadmat(savedir / f'ICASSO_{string}_{ncomps}_results.mat', variable_names=['Iq', 'S'])
    iq = result['Iq'].ravel()

    pattern = result['S'].T
    pattern_reliable = pattern[iq > 0.8]
    savemat(savedir / 'reliable_state_pattern.mat', {'pattern': pattern_reliable})
    print(f'Saved {len(pattern_reliable)} reliable patterns with Iq > 0.8.')


if __name__ == '__main__':
    sample_dir = Path(__file__).resolve().parent
    network_timeseries_list = [
        sample_dir / 'timeseries_samples' / 'area_timeseries_1.mat',
        sample_dir / 'timeseries_samples' / 'area_timeseries_2.mat',
    ]
    TR_list = [0.72, 0.72]

    decompose_dFNC(network_timeseries_list, TR_list, savedir='.', string='sample')
