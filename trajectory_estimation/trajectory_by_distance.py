import numpy as np
from scipy.io import loadmat, savemat

def high_low_confidence_range(data, confidence=[2.5, 97.5]):

    lower = np.percentile(data, confidence[0])
    upper = np.percentile(data, confidence[1])
    mean = np.mean(data)
    return  upper, lower, mean

distance_matrix = np.load('BCP_Neuromark53_area_Eurodistance.npy')
distance_matrix = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]


def population_distance_trajectory():
    CR_bootstrap_list = []
    for trail in range(10):
        traj = np.load(f'./bootstrap/CS_conn_derivate_traj_{trail}.npy')*1000
        # traj_mean = abs(traj).mean(axis=0)
        CR_bootstrap_list.append(traj)
    growth_rate_traj = np.array(CR_bootstrap_list).mean(axis=0)

    total_CR_conf_list = {}
    total_CR_mean_list = {}
    for i in range(1,11):
        indice = (distance_matrix<(i*10+10)) & ((distance_matrix>=(i*10)))
        tmp_list = []
        for trail in range(10):
            traj = np.load(f'./bootstrap/CS_conn_derivate_traj_{trail}.npy')*1000
            tmp_list.append(traj[indice].mean(axis=0))
        tmp_list = np.array(tmp_list)
        CR_conf_list = []
        CR_mean_list = []
        for t in range(tmp_list.shape[1]):
            high, low, mean = high_low_confidence_range(tmp_list[:, t], [2.5, 97.5])
            CR_conf_list.append([high, low])
            CR_mean_list.append(mean)
        total_CR_conf_list.update({i: np.array(CR_conf_list)})
        total_CR_mean_list.update({i: np.array(mean)})
    savemat('mean_growth_rate_traj.mat',{'CR_traj': growth_rate_traj})
    savemat('mean_growth_rate_dis_mean_traj.mat',{'CR_traj': total_CR_mean_list})
    savemat('mean_growth_rate_dis_conf_traj.mat',{'CR_traj': total_CR_conf_list})