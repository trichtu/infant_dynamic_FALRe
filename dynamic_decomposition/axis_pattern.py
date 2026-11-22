import numpy as np
from scipy.io import savemat, loadmat

def axe_pattern():
    state_pattern = loadmat('component_5_pattern_pos.mat')['components']
    n_edges = state_pattern.shape[1]
    n_networks = (1 + int(np.sqrt(1 + 8 * n_edges))) // 2
    rows, cols = np.triu_indices(n_networks, k=1)
    vector_list = []
    for i in range(5):
        matrix = np.zeros((n_networks, n_networks))
        matrix[rows, cols] = state_pattern[i]
        matrix[cols, rows] = state_pattern[i]
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        vector = eigenvectors[:, np.argmax(np.abs(eigenvalues))]
        vector_list.append(vector / vector.std())
        
    vector_list = np.array(vector_list)
    print(vector_list.shape)
    savemat('./eigenvector.mat',{'components': vector_list})