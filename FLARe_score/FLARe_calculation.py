import numpy as np
from pathlib import Path
from scipy.io import loadmat, savemat
from run_constrain_ica import run_constrain_ica

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


def calculate_asymmetric_features(signal):
    
    """
    Parameters:
        signal (array): Input time series.
 
    Returns:
        dict: A dictionary containing the calculated features.
    """

    positive_peaks = signal[signal > 0]
    negative_peaks = signal[signal < 0]

    if len(positive_peaks) > 0 and len(negative_peaks) > 0:
        fluctuation_pos = np.mean(positive_peaks**2)
        fluctuation_neg = np.mean(negative_peaks**2)
        fluctuation_ratio = np.log(fluctuation_pos/fluctuation_neg)

    else:
        fluctuation_ratio = np.nan
        
    return fluctuation_ratio


def calculate_FLARe(network_timeseries, state_patterns, TR, savestring='test', sliding_window=45, save_dir= '.'):

    n_edges = loadmat(state_patterns)['components'].shape[1]
    n_networks = (1 + int(np.sqrt(1 + 8 * n_edges))) // 2
    if network_timeseries.shape[1] != n_networks and network_timeseries.shape[0] == n_networks:
        network_timeseries = network_timeseries.T
    if network_timeseries.shape[1] != n_networks:
        raise ValueError('Network count does not match the state patterns.')
    if not 2 <= np.ceil(sliding_window / TR) <= len(network_timeseries):
        raise ValueError('The sliding window must contain at least two scans and fit in the time series.')

    matrices = dynamic_corr([network_timeseries], [TR], slide_window=sliding_window)[0]
    rows, cols = np.triu_indices(n_networks, k=1)
    edges = matrices[:, rows, cols].T
    demeaned_edges = edges - edges.mean(axis=1, keepdims=True)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dFNC_path = save_dir / f'{savestring}_dFNC.mat'
    savemat(dFNC_path, {'timeseries': demeaned_edges})
    state_timeseries = run_constrain_ica(dFNC_path, save_dir, state_patterns, savestring)
    return [calculate_asymmetric_features(series) for series in state_timeseries.T]


if __name__ == '__main__':
    sample_dir = Path(__file__).resolve().parent
    network_timeseries = loadmat(sample_dir / 'sample_53_network_timeseries.mat')['timeseries']
    state_patterns = sample_dir / 'component_5_pattern_pos.mat'
    print('FLARe scores:', calculate_FLARe(network_timeseries, state_patterns, TR=0.72, savestring='state_test'))
