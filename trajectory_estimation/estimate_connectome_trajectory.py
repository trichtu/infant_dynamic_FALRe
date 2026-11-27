import numpy as np
from scipy.io import savemat


save_dir = '.'

def fisher_z_transform(corr_matrix):
    """
    Applies the Fisher z-transformation to a correlation matrix.

    Parameters:
    corr_matrix (numpy.ndarray): Square correlation matrix of shape (n, n)

    Returns:
    numpy.ndarray: Transformed matrix with Fisher z-scores
    """
    # Ensure the matrix is a numpy array
    corr_matrix = np.array(corr_matrix)
    corr_matrix[np.isnan(corr_matrix)] = 0
    np.fill_diagonal(corr_matrix, 0)

    # Apply Fisher transformation, avoiding invalid values like r=1 or r=-1
    np.seterr(divide='ignore', invalid='ignore')  # Handle potential warnings for divide by zero

    fisher_z = 0.5 * np.log((1 + corr_matrix) / (1 - corr_matrix))
    fisher_z[np.isinf(fisher_z)] = 0
    
    return fisher_z

def rest_FNC(timeseries_list):
    FNC_list = []
    for timeseries in timeseries_list:
        corr = np.corrcoef(timeseries.T)
        corr = fisher_z_transform(corr)
        FNC_list.append(corr)
    FNC_list = np.array(FNC_list)
    return FNC_list

def calculate_modularity_matrix(matrix):
    domain_list = ['SCN', 'AUD', 'SMN', 'VIS', 'CON', 'DMN', 'CER' ]
    domain_position = np.array([[0,5],[5,7],[7, 16],[16, 25], [25, 42], [42, 49], [49, 53]])

    mean_modularity = np.zeros([7,7])
    for i, domain1 in enumerate(domain_list):
        for j, domain2 in enumerate(domain_list):
            tmp = matrix[ domain_position[i][0]:domain_position[i][1], domain_position[j][0]:domain_position[j][1]]
            if i==j:
                mean = tmp[ np.triu_indices(tmp.shape[1], k=1)].mean()
            else:
                mean = tmp.mean()
            mean_modularity[i,j] = mean

    mean_cross_modular = list(mean_modularity[np.triu_indices(mean_modularity.shape[1], k=0)])
    return np.array(mean_cross_modular)



def estimate_growth_trajectory(n_splines=10, n_boot=None, x1_pred=None):
    """Fit the first 10 connections using the bundled connectivity_example.npz.

    The example retains all scans, aligned age/sex/motion/site covariates,
    infant IDs, and the existing resample masks. n_boot selects how many
    saved resamples to fit (all by default).
    """
    from pathlib import Path
    if __package__:
        from .GAMM_function import GAMM_trajectory_estimate
    else:
        from GAMM_function import GAMM_trajectory_estimate

    input_dir = Path(__file__).resolve().parent
    output_dir = Path(save_dir) / 'bootstrap'
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(input_dir / 'connectivity_example.npz') as example:
        indices = example['resample_indices']
        connectivity = example['connectivity']
        subjects = example['subject']
        ages = example['age']
        genders = example['gender']
        motion = example['motion']
        sites = example['site']
    if x1_pred is None:
        x1_pred = np.arange(25, 176, dtype=float)

    for number in range(n_boot):
        unpaired_indice = indices[number]
        subject_id = subjects[unpaired_indice]
        static_connectivity = connectivity[unpaired_indice]
        unpaired_age_list = ages[unpaired_indice]
        unpaired_gender_list = genders[unpaired_indice]
        unpaired_hd_list = motion[unpaired_indice]
        unpaired_site_list = sites[unpaired_indice]
        print('connectivity resample:', number, 'features:', static_connectivity.shape)
        conn_derivative_list, conn_traj_list, conn_confid_list = [], [], []
        for i in range(static_connectivity.shape[1]):
            print(i)
            _, conn_traj, conf_conn, conn_derivative = GAMM_trajectory_estimate(
                static_connectivity[:, i], unpaired_age_list, unpaired_gender_list,
                unpaired_hd_list, unpaired_site_list, subject_id,
                gradients=True, x1_pred=x1_pred, n_splines=n_splines)
            conn_derivative_list.append(conn_derivative)
            conn_traj_list.append(conn_traj)
            conn_confid_list.append(conf_conn)

        np.save(output_dir / f'CS_list_dataset_{number}.npy', np.array(conn_traj_list))
        savemat(output_dir / f'CS_derivative_dataset_{number}.mat',
                {'dFNC': np.array(conn_derivative_list)})
        

if __name__ == '__main__':
    estimate_growth_trajectory()


    
