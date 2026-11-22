import numpy as np
from scipy.io import savemat, loadmat
from pathlib import Path
from run_constrain_ica import run_constrain_ica



def calculate_axe_amplitude(network_timeseries, state_eigenvector, TR, savestring, save_dir='.', window=45):
    axe_timeseries = run_constrain_ica(network_timeseries, save_dir, state_eigenvector, savestring)
    window_scans = int(np.ceil(window / TR))
    if not 1 <= window_scans <= len(axe_timeseries):
        raise ValueError('The amplitude window must fit in the time series.')
    window_count = len(axe_timeseries) - window_scans + 1
    smoothed_amplitudes = []
    for start in range(window_count):
        window_data = axe_timeseries[start:start + window_scans]
        mean_amplitude = np.abs(window_data).mean(axis=0)
        smoothed_amplitudes.append(mean_amplitude)

    smoothed_amplitudes = np.array(smoothed_amplitudes)
    return smoothed_amplitudes



if __name__ == '__main__':

    sample_dir = Path(__file__).resolve().parent
    state_eigenvector = sample_dir / 'eigenvector.mat'
    network_timeseries = sample_dir / 'sample_53_network_timeseries.mat'
    axe_amplitude_sm = calculate_axe_amplitude(network_timeseries, state_eigenvector, TR=0.72, savestring='axe_test')
    np.save('axe_test_amplitudes.npy', axe_amplitude_sm)
    print('Saved axis amplitudes:', axe_amplitude_sm.shape)
