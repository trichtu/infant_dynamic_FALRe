import numpy as np
from scipy.io import savemat, loadmat

def axe_pattern():
    state_pattern = loadmat('./dynamic_volumne/ICA_results/component_5_pattern_pos.mat')['components']
    print(state_pattern.shape)
    vector_list = []
    for i in range(5):
        eigenvalues, eigenvectors = np.linalg.eig(state_pattern[i])
        map = eigenvectors[:, 0]
        vector_list.append(map/map.std())
        
    vector_list = np.array(vector_list)
    print(vector_list.shape)
    savemat('./dynamic_volumne/script/eigenvector.mat',{'components': vector_list})


def smoothed_axe_amplitude(axe_timeseries, window=45, TR=0.72):
    slice = int(np.ceil(window/TR))
    axe_amplitude_sm = []                      
    for i in range(len(axe_timeseries)-slice+1): 
        window_data = np.abs(axe_timeseries[i:i+slice,:]).mean(axis=0)
        axe_amplitude_sm.append(window_data)
    axe_amplitude_sm =  np.array(axe_amplitude_sm)
    return axe_amplitude_sm

