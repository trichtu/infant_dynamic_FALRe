import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, l
from scipy.stats import pearsonr
from scipy.stats import pearsonr
from scipy.io import loadmat
import statsmodels.stats.multitest as smm 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from scipy.io import savemat
from infant_dataset import plot_FNC, session_infomation_whole
from spontaneous_dynamic_decomposition import back_upper_triangles
from scipy.stats import ttest_ind


save_dir = './dynamic_volumne/script/GAM_resutls'


def calculate_weights(session_list, paired=True):
     
    if paired:
        session_list = np.array([ session[0]+session[1] for session in session_list])

    unique_session = np.array(list(set(session_list))) 
    count_list = []
    for session in unique_session:
        counts = (session_list==session).sum()
        count_list.append(counts)
    count_list = np.array(count_list)

    weight_list = []
    for session in session_list:
        counts = count_list[unique_session==session][0]
        weights = 1.0/counts
        weight_list.append(weights)

    return np.array(weight_list)


def compute_bic(gam, X, y):
    n = len(y)
    k = gam.statistics_['edof']  # effective degrees of freedom
    logL = gam.loglikelihood(X, y)
    bic = np.log(n)*k - 2*logL
    return bic

def GAM_trajectory_estimate(target, age_list, gender_list, hd_list, site_list, weights='None', gradients=False, x1_pred = np.linspace(0, 180, 181), grid_search=False, lam_default=10):
    if np.array(age_list).ndim==1:
        age_list = np.expand_dims(age_list, 1)
        gender_list = np.expand_dims(gender_list, 1)
        hd_list = np.expand_dims(hd_list, 1)     
        site_list = np.expand_dims(site_list, 1)  
       
    index = ~np.isnan(target)
    target = target[index]
    age_list = age_list[index]
    gender_list, hd_list, site_list = gender_list[index], hd_list[index], site_list[index]
    print(age_list.shape, gender_list.shape, hd_list.shape, site_list.shape)
    X = np.concatenate([age_list, gender_list, hd_list, site_list], axis=1)
    print('age range', age_list.max(), age_list.min())

    # Define a simple GAM model
    gam_search = LinearGAM(s(0, n_splines=10) + l(1) + l(2)+ l(3))


    # # Perform grid search
    # Fit a GAM model with a smooth function of x1z
    if grid_search:
        # lam_values =  np.logspace(-3, 3, 10) 
        lam_values = np.array([1, 5]+list(np.logspace(1, 3, 10)))
        
        weights = weights[index]

        if not isinstance(weights, str):
            gam_search.gridsearch(X, target, lam=[lam_values, [0], [0],[0]], weights=weights)
        else:
            gam_search.gridsearch(X, target, lam=[lam_values, [0], [0],[0]])
   
        lam = gam_search.lam[0]
        print('lam', gam_search.lam)

    else:
        lam = lam_default
        
    if not isinstance(weights, str):
        weights = weights[index]
        gam = LinearGAM(s(0, n_splines=10, lam=lam) + l(1) + l(2) + l(3)).fit(X, target, weights=weights)
    else:    
        gam = LinearGAM(s(0, n_splines=10, lam=lam) + l(1) + l(2) + l(3)).fit(X, target)

    XX1 = np.column_stack((x1_pred, gender_list, hd_list, site_list ))
    y_pred = gam.predict(XX1)
    conf_int = gam.prediction_intervals(XX1, width=0.975)

    # Compute the approximate derivative (change rate) of y with respect to x1 using finite differences
    if gradients:
        y_derivative = np.gradient(y_pred, x1_pred)
        return gam, y_pred, conf_int, y_derivative
    else:
        return gam, y_pred, conf_int

    
def calculate_within_cross_domain_index(Xdomain, Ydomain):
    domain_list = ['SC', 'AUD', 'SM', 'VS', 'CC', 'DM', 'CB' ]
    domain_position = np.array([[0,5],[5,7],[7, 16],[16, 25], [25, 42], [42, 49], [49, 53]])
    domain_dict = {}
    for d in range(7):
        for j in range(domain_position[d][0], domain_position[d][1]):
            domain_dict.update({j: domain_list[d] })
    index = []
    x, y = np.triu_indices(53, k=1)
    for i in range(len(x)):
        x_domain = domain_dict[x[i]]
        y_domain = domain_dict[y[i]]
        if  (x_domain in Xdomain) & (y_domain in Ydomain) :
            index.append(i)
        elif (x_domain in Xdomain) & (y_domain in Ydomain) :
            index.append(i)
 
    index = np.array(index)

    return index



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


def estimate_growth_rate_trajectory(string = '' ):
    save_dir = './dynamic_volumne/script/GAM_resutls'
    state_number = 5
    
    for study in [  'all']:
        # independent variables
        # [sample_number]
        session_list = np.load(f'./dynamic_volumne/script/session_list_{state_number}{string}.npy')[:, 0]
        study_list = session_infomation_whole(session_list, 'DV_dataset')
        if study in [1, 2]:
            indice = study_list == study
        else:
            indice = np.ones_like(study_list).astype(bool)
        paired_session_list = np.load(f'./dynamic_volumne/script/session_list_{state_number}{string}.npy')[indice]
        weights = calculate_weights(paired_session_list, paired=True)

        # [sample_number, connection_number]
        change_list = np.load(f'./dynamic_volumne/script/change_list_{state_number}{string}.npy')[indice]
        base_age_list =  np.load(f'./dynamic_volumne/script/demo_list_{state_number}{string}.npy')[indice, 0:1]
        base_age_list = base_age_list*7
        time_interval = np.load(f'./dynamic_volumne/script/demo_list_{state_number}{string}.npy')[indice, 1:2]
        gender_list = np.load(f'./dynamic_volumne/script/demo_list_{state_number}{string}.npy')[indice, 2:3]
        hd_list = np.load(f'./dynamic_volumne/script/demo_list_{state_number}{string}.npy')[indice, 4:5]
        site_list = np.load(f'./dynamic_volumne/script/site_list_{state_number}{string}.npy').reshape(-1,1)
        time_interval = time_interval*7

        mid_age_list = base_age_list + 0.5*time_interval
        # print(time_interval.shape, state_feature.shape, change_list.shape)
        change_connectome = change_list / time_interval 
        print('paired change shape:', change_connectome.shape)
        print('paired age range:', mid_age_list.min(), mid_age_list.max())

        # change rate trajectory
        connection_number = change_list.shape[1]
        change_derivative_list = []
        change_rate_list = []
        confidence_list = []
        print(gender_list.shape, hd_list.shape, site_list.shape)
        for i in range(connection_number):
            _, change_rate,conf_change_rate, change_rate_derivative = GAM_trajectory_estimate(change_connectome[:, i], mid_age_list, gender_list, hd_list, site_list, weights, gradients=True, grid_search=True)
            change_derivative_list.append(change_rate_derivative)
            confidence_list.append(conf_change_rate)
            change_rate_list.append(change_rate)
        change_rate_list = np.array(change_rate_list)
        confidence_list = np.array(confidence_list)
        change_derivative_list = np.array(change_derivative_list)
        np.save(f'{save_dir}/CA_list_dataset_{study}.npy', change_derivative_list)
        np.save(f'{save_dir}/CR_list_dataset_{study}.npy', change_rate_list)
        np.save(f'{save_dir}/CR_confidence_list_dataset_{study}.npy', confidence_list )
    
        _, change_rate, conf_change_rate, change_rate_derivative = GAM_trajectory_estimate( (change_connectome).mean(axis=1), mid_age_list, gender_list, hd_list, weights, gradients=True) #, grid_search=False, lam_default=lam_default)
        np.save(f'{save_dir}/CA_single_dataset_{study}.npy',  np.array(change_rate_derivative))
        np.save(f'{save_dir}/CR_single_dataset_{study}.npy', np.array(change_rate))
        np.save(f'{save_dir}/CR_confidence_single_dataset_{study}.npy', np.array(conf_change_rate) )

        _, change_rate, conf_change_rate, change_rate_derivative = GAM_trajectory_estimate( abs(change_connectome).mean(axis=1), mid_age_list, gender_list, hd_list, weights, gradients=True) #, grid_search=False, lam_default=lam_default)
        np.save(f'{save_dir}/CA_singleabs_dataset_{study}.npy',  np.array(change_rate_derivative))
        np.save(f'{save_dir}/CR_singleabs_dataset_{study}.npy', np.array(change_rate))
        np.save(f'{save_dir}/CR_confidence_singleabs_dataset_{study}.npy', np.array(conf_change_rate) )


        modular_change = np.array([calculate_modularity_matrix(back_upper_triangles(change, 53, k=1)) for change in change_connectome])
        domain_list = ['SC', 'AUD', 'SM', 'VS', 'CC', 'DM', 'CB' ]
        for i in range(7):
            _, change_rate, conf_change_rate, change_rate_derivative = GAM_trajectory_estimate( modular_change[:, i], mid_age_list, gender_list, hd_list, weights, gradients=True) #, grid_search=False, lam_default=lam_default)
            np.save(f'{save_dir}/CR_{domain_list[i]}_dataset_{study}.npy', np.array(change_rate))
            np.save(f'{save_dir}/CR_confidence_{domain_list[i]}_dataset_{study}.npy', np.array(conf_change_rate) )


    return None


def estimate_growth_trajectory():
    for study in ['all']:
        # unpaired state feature 
        unpaired_session_list = np.load(f'unpaired_session_list_5.npy')
        unpaired_study_list = session_infomation_whole(unpaired_session_list, 'DV_dataset')
        if study in [1, 2]:
            unpaired_indice = unpaired_study_list == study
        else:
            unpaired_indice = np.ones_like(unpaired_study_list).astype(bool)

        session_list = np.load(f'unpaired_session_list_5.npy')[unpaired_indice]
        weights = calculate_weights(session_list, paired=True)

        # [sample_number, state_number]
        static_connectivity = np.load(f'unpaired_sFNC_list_5.npy')[ unpaired_indice]
        print('connectivity feature:', static_connectivity.shape)
        unpaired_age_list = np.load(f'unpaired_demo_list_5.npy')[unpaired_indice, 2]
        unpaired_age_list = unpaired_age_list*7
        unpaired_gender_list = np.load(f'unpaired_demo_list_5.npy')[unpaired_indice, 0]
        unpaired_hd_list =  np.load(f'unpaired_demo_list_5.npy')[unpaired_indice, 6]
        unpaired_TR_list = np.load(f'unpaired_demo_list_5.npy')[unpaired_indice, 5]
        unpaired_site_list = np.array([1 if abs(vv-0.72)<0.001 else 0 for vv in unpaired_TR_list])
        # unpaired_site_list = np.ones_like(unpaired_site_list)
        print('unpaired age range:', unpaired_age_list.min(), unpaired_age_list.max() )
        print(unpaired_age_list.shape, unpaired_gender_list.shape, unpaired_hd_list.shape, unpaired_site_list.shape)

        # state trajectory 
        conn_feature_number = static_connectivity.shape[1]
        conn_derivative_list = []
        conn_traj_list = []
        conn_confid_list = []
        
        for i in range(conn_feature_number):
            # print('conn',static_connectivity[:, i].shape, unpaired_age_list.shape, unpaired_gender_list.shape, unpaired_hd_list.shape)
            _, conn_traj, conf_conn, conn_derivative = GAM_trajectory_estimate(static_connectivity[:, i], unpaired_age_list, unpaired_gender_list, unpaired_hd_list, unpaired_site_list, weights, gradients=True, grid_search=True)
            conn_derivative_list.append(conn_derivative)
            conn_traj_list.append(conn_traj)
            conn_confid_list.append(conf_conn)
        
        conn_derivative_list = np.array(conn_derivative_list)
        conn_traj_list = np.array(conn_traj_list)
        conn_confid_list = np.array(conn_confid_list)
        
        np.save(f'{save_dir}/CS_list_dataset_{study}.npy', conn_traj_list)
        np.save(f'{save_dir}/CS_conf_list_dataset_{study}.npy',  conn_confid_list)
        np.save(f'{save_dir}/CS_derivative_dataset_{study}.npy', conn_derivative_list)
        
        savemat(f'{save_dir}/CS_derivative_dataset_{study}.mat', {'dFNC': conn_derivative_list})
        
        _, conn_traj, conf_conn, conn_derivative = GAM_trajectory_estimate((static_connectivity).mean(axis=1), unpaired_age_list, unpaired_gender_list, unpaired_hd_list, weights, gradients=True) #, grid_search=False, lam_default=lam_default)
        np.save(f'{save_dir}/CS_single_dataset_{study}.npy', np.array(conn_traj))
        np.save(f'{save_dir}/CS_conf_single_dataset_{study}.npy',  np.array(conf_conn))
        np.save(f'{save_dir}/CS_derivative_single_dataset_{study}.npy', np.array(conn_derivative))        
        
        _, conn_traj, conf_conn, conn_derivative = GAM_trajectory_estimate(abs(static_connectivity).mean(axis=1), unpaired_age_list, unpaired_gender_list, unpaired_hd_list, weights, gradients=True) #, grid_search=False, lam_default=lam_default)
        np.save(f'{save_dir}/CS_singleabs_dataset_{study}.npy', np.array(conn_traj))
        np.save(f'{save_dir}/CS_conf_singleabs_dataset_{study}.npy',  np.array(conf_conn))
        np.save(f'{save_dir}/CS_derivative_singleabs_dataset_{study}.npy', np.array(conn_derivative))  


        modular_change = np.array([calculate_modularity_matrix(back_upper_triangles(change, 53, k=1)) for change in static_connectivity])
        domain_list = []
        domain_deri_list = []
        for i in range(modular_change.shape[1]):
            _, CS_traj, conf_CS, CS_derivative = GAM_trajectory_estimate( modular_change[:, i], unpaired_age_list, unpaired_gender_list, unpaired_hd_list, weights, gradients=True)
            domain_list.append(CS_traj)
            domain_deri_list.append(CS_derivative)
        domain_list = np.array(domain_list)
        domain_deri_list = np.array(domain_deri_list)
        np.save(f'{save_dir}/CS_domain_list.npy', domain_list)
        np.save(f'{save_dir}/CS_derivative_domain_list.npy', domain_deri_list)
        
        
def estimate_state_trajectory(state_number = 5, string = ''):
    save_dir='./dynamic_volumne/script/GAM_resutls/'
    # for lam in [5]: #[5, 10, 15, 20, 25, 30, 35, 40]:
    for study in ['all']:
        # unpaired state feature 
        unpaired_session_list = np.load(f'unpaired_session_list_{state_number}{string}.npy')
        unpaired_study_list = session_infomation_whole(unpaired_session_list, 'DV_dataset')
        unpaired_indice = np.ones_like(unpaired_study_list).astype(bool)

        session_list = np.load(f'unpaired_session_list_{state_number}{string}.npy')[unpaired_indice]
        weights = calculate_weights(session_list, paired=False)

        state_feature = np.load(f'unpaired_state_asymetry_5_duration.npy')[unpaired_indice]
        unpaired_age_list = np.load(f'unpaired_demo_list_5.npy')[unpaired_indice, 2]
        unpaired_age_list = unpaired_age_list*7
        unpaired_gender_list = np.load(f'unpaired_demo_list_5.npy')[unpaired_indice, 0]
        unpaired_hd_list =  np.load(f'unpaired_demo_list_5.npy')[unpaired_indice, 6]
        unpaired_TR_list = np.load(f'unpaired_demo_list_5.npy')[unpaired_indice, 5]
        unpaired_site_list = np.array([1 if abs(vv-0.72)<0.001 else 0 for vv in unpaired_TR_list])
        unpaired_site_list = np.zeros_like(unpaired_site_list)
        print('unpaired state feature shape:', state_feature.shape)
        print('unpaired age range:',unpaired_age_list.shape, unpaired_age_list.min(), unpaired_age_list.max() )

        # state trajectory d
        state_feature_number = state_feature.shape[1]
        state_derivative_list = []
        state_traj_list = []
        for i in range(state_feature_number):
            print('state', i)
            _, state_traj, conf_state, state_feature_derivative = GAM_trajectory_estimate(state_feature[:, i], unpaired_age_list, unpaired_gender_list, unpaired_hd_list, unpaired_site_list, weights, gradients=True, grid_search=False, lam_default=100) # gradients=True, grid_search=True) 
            state_derivative_list.append(state_feature_derivative)
            state_traj_list.append(state_traj)

        state_traj_list = np.array(state_traj_list)
        np.save(f'{save_dir}/FLARe_state_traj_list_dataset_{study}_cluster_{state_number}_duration.npy', state_traj_list)
        
    return None



def resample_dataset(percent=0.8, state_number = 5, string = '' ):

    # unpaired information
    unpaired_session = np.load(f'unpaired_session_list_{state_number}{string}.npy')
    unpaired_subid = np.array([ sess[:8] for sess in unpaired_session])
    unpaired_unique_subid = np.array(list(set(unpaired_subid)))
    unpaired_gender = session_infomation_whole(unpaired_session, 'gender')
    unpaired_age = session_infomation_whole(unpaired_session, 'corrected_age_week')

    # paried information
    paired_demo = np.load(f'demo_list_{state_number}{string}.npy')
    paired_age =  paired_demo[:, 0] + 0.5*paired_demo[:, 1]
    paired_gender = paired_demo[:, 2]
    paried_session =  np.load(f'session_list_{state_number}{string}.npy')
    paired_subid = np.array([ sess[:8] for sess in paried_session[:,0] ])
    paired_unique_sub_id = np.array(list(set(paired_subid)))

    print(len(unpaired_unique_subid), len(paired_unique_sub_id))
    optimal_max = 0
    subject_resample_number = int(len(unpaired_unique_subid)*percent) 
    dataset_list = []

    np.random.shuffle(unpaired_unique_subid)
    
    dataset_1 = unpaired_unique_subid[:subject_resample_number]
    # dataset_2 = unpaired_unique_subid[35:]
    dataset_list.append(unpaired_unique_subid)
    
    unpaired_index_for_dataset_1 = np.array([ subid in dataset_1 for subid in unpaired_subid])
    paired_index_for_dataset_1 = np.array([ subid in dataset_1 for subid in paired_subid])

    return unpaired_index_for_dataset_1, paired_index_for_dataset_1


def bootstrap_for_reliable(state_number=5, string=''):

    growth_rate_replicate_list = []
    for i in range(100):
        print('times', i)
        unpaired_indice, paried_indice = resample_dataset( percent=0.8)
        session_list = np.load(f'./dynamic_volumne/script/session_list_{state_number}{string}.npy')[paried_indice, 0]

        paired_session_list = np.load(f'./dynamic_volumne/script/session_list_{state_number}{string}.npy')[paried_indice]
        weights = calculate_weights(paired_session_list, paired=True)

        # [sample_number, connection_number]
        change_list = np.load(f'./dynamic_volumne/script/change_list_{state_number}{string}.npy')[paried_indice]
        base_age_list =  np.load(f'./dynamic_volumne/script/demo_list_{state_number}{string}.npy')[paried_indice, 0:1]
        time_interval = np.load(f'./dynamic_volumne/script/demo_list_{state_number}{string}.npy')[paried_indice, 1:2]
        
        base_age_list = base_age_list*7
        time_interval = time_interval*7
        mid_age_list = base_age_list + 0.5*time_interval

        # change rate trajectory
        connection_number = change_list.shape[1]
        change_connectome = change_list / time_interval 
        print(change_connectome.shape, mid_age_list.min(), mid_age_list.max())
        change_rate_list = []
        for i in range(connection_number):
            _, change_rate = GAM_trajectory_estimate(change_connectome[:, i], mid_age_list, weights, gradients=False,  lam=30) 
            change_rate_list.append(change_rate)
            np.save(f'{save_dir}/bootsgrap_growth_rate_tra_{i}.npy', change_rate_list)  
        change_rate_list = np.array(change_rate_list)
        growth_rate_replicate_list.append(change_rate_list)

    return None



if __name__ == '__main__':


    estimate_growth_rate_trajectory()
    estimate_growth_trajectory()
    estimate_state_trajectory()