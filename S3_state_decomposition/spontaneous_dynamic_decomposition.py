import numpy as np
import pandas as pd
import nibabel as nib
from datetime import datetime
from infant_dataset import dataset_info, sample_sponta_dynamic_dataset, session_infomation_whole
from infant_dataset import load_ICA_data, dynamic_corr, rest_FNC
import mne
from scipy.signal import hilbert
# from mne.preprocessing import ICA
from scipy.io import loadmat, savemat
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from data_loader import get_longitudinal_infant_dataset, get_evaluation_loader
import os
from sklearn.metrics import silhouette_score
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from scipy.stats import skew
import networkx as nx
import community as community_louvain
import bct

ICA_result_dir='./dynamic_volumne/ICA_results'


def back_upper_triangles(upper_triangle, size, k=1):
    upper_triangle = np.array(upper_triangle)

    if upper_triangle.ndim==1:
        # Create an empty matrix filled with zeros
        reconstructed_matrix = np.zeros((size, size)).astype('float')

        # Fill the upper triangular part (including diagonal) of the matrix
        reconstructed_matrix[np.triu_indices(size, k=k)] = upper_triangle
        reconstructed_matrix = reconstructed_matrix + reconstructed_matrix.T

    elif upper_triangle.ndim==2:
        reconstructed_matrix = []
        for i in range(upper_triangle.shape[0]):
            tmp_mat = np.zeros((size, size)).astype('float')
            tmp_mat[np.triu_indices(size, k=k)] = upper_triangle[i, :]
            tmp_mat = tmp_mat + tmp_mat.T
            reconstructed_matrix.append(tmp_mat)
        reconstructed_matrix = np.array(reconstructed_matrix)
    return reconstructed_matrix


def match_component(single_component, component_template):
    """
    single_component shape: [Feature_number,1] 
    component_template shape: [Feature_number, N]
    """
    comps = component_template.shape[1]
    corr_list = []
    for i in range(comps):
        corr = np.corrcoef(single_component, component_template[:,i])[0,1]
        corr_list.append(corr)
    # print(corr_list)
    component_indice = np.argmax(np.abs(corr_list))
    corrmax = corr_list[component_indice]
    return corrmax, component_indice


def choose_centrelized_components(number_best, trail_times, plot=False):
    
    target = f'{ICA_result_dir}/component_{number_best}_pattern.mat'
    if True: # not os.path.exists(target):
        trail_list = []
        for trail_number in range(trail_times):
            component = loadmat(f'{ICA_result_dir}/infomax_components_{number_best}_trail_{trail_number}.mat')['components']
            print(trail_number, 'loading best components:', component.shape)
            component = component.reshape(-1)
            trail_list.append(component)
        corr = np.corrcoef(trail_list)
        mean_corr = corr.mean(axis=0)
        trail_best = np.argmax(mean_corr)
        component = loadmat(f'{ICA_result_dir}/infomax_components_{number_best}_trail_{trail_best}.mat')['components']
        component = component/component.std(axis=1, keepdims=True)
        savemat(f'{ICA_result_dir}/component_{number_best}_pattern_old.mat', {'components': component})

        sys = back_upper_triangles(component, 53)
        sys = np.array([mat + mat.T for mat in sys])
        savemat(f'{ICA_result_dir}/component_{number_best}_pattern_sys.mat', {'components': sys})

    return component


def plot_state_pattern(state_number, trans=False, vmin='None', vmax='None'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.io import loadmat
    import pandas as pd
    component_priors = pd.read_excel('./MatchTable_High_NetworkLabling_Finalized.xlsx')
    labels = component_priors['Domain (with hippocampus)'].values[:53] # python begin at 0

    pattern = loadmat(f'{ICA_result_dir}/component_{state_number}_pattern_sys.mat')['components']
    print(pattern.shape)
    scale = max(abs(pattern.min()), pattern.max())
    for ii,matrix in enumerate(pattern):
        # Generate a random 53x53 matrix
        if trans:
            matrix = matrix.T + matrix
            np.fill_diagonal(matrix, 1)
        # Create 53 labels for the x and y axis
        # labels = [f'Label {i+1}' for i in range(53)]

        # Create a heatmap using seaborn
        plt.figure(figsize=(10, 8))

        if isinstance(vmin, str):
            ax = sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap='bwr', vmin=-scale , vmax=scale, annot=False) # 'RdBu'
        else:
            ax = sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap='bwr', vmin=vmin , vmax=vmax, annot=False)
        
        domain_labelset = ['SCN', 'AUD', 'SMN', 'VIS', 'CON', 'DMN', 'CER']
        domain_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan']
        y_position = [ 2.5,  6, 12.5,  20.5,  33.5,  45.5,  51 ]
        plt.yticks(ticks=y_position, labels=domain_labelset,  ha='right', size=15)

        rect_position = np.array([[0,4],[5,6],[7, 15],[16, 24], [25, 41], [42, 48], [49, 52]])
        for i in range(len(domain_labelset)):
            # Rectangle(x, y, width, height)
            plt.gca().add_patch(Rectangle((-5, rect_position[i][0]), 5, rect_position[i][1]-rect_position[i][0]+1, color=domain_colors[i], clip_on=False))  

        x_position = np.array([ 2.5,  6, 12.5,  20.5,  33.5,  45.5,  51 ])
        plt.xticks(ticks=x_position+1, labels=domain_labelset,  ha='right', rotation=90, size=15)
        for i in range(len(domain_labelset)):
            plt.gca().add_patch(Rectangle((rect_position[i][0], 53 ), rect_position[i][1]-rect_position[i][0]+1,  5, color=domain_colors[i], clip_on=False))  

        print(f'{ICA_result_dir}/pattern_{state_number}_{ii}.svg')
        plt.savefig(f'{ICA_result_dir}/pattern_{state_number}_{ii}.svg', dpi=1000)

    return None


def regression_covariations(features, covariates):
    """
    feature shape: [subjects, feature]
    covariate shape: [subjects, variates]
    """
    from sklearn.linear_model import LinearRegression
    residuals = np.zeros_like(features) 
    N = features.shape[1]
    for i in range(N):
        regressor = LinearRegression()
        # Regress covariates from each feature column
        regressor.fit(covariates, features[:, i])
        
        # Predicted values of the feature based on covariates
        predicted = regressor.predict(covariates)
        # Residual = original feature - predicted (covariate-driven part)
        residuals[:, i] = features[:, i] - predicted

    return residuals

	
def paired_state_multifeatures(state_number = 5):
    
    dataset = get_longitudinal_infant_dataset('all', state_number, dataset_name='longitudinal_infant', \
                                                                    timepoint_number='all', TD_only=True)
    dataloader = get_evaluation_loader(dataset, 1, 1)
    paired_session_list = dataset.all_paired_session
    # baseline_state_timeseries = dataset.baseline_state_timeseries
    # afterline_state_timeseries = dataset.afterline_state_timeseries
    baseline_list = []
    after_list = []
    change_list = []
    demo_list= []
    session_infor_list = []
    base_asymmetry_list = []
    after_asymmetry_list = []
    site_list = []
    ee = 0
    for i, (base_state_path, after_state_path, demo, base_connectome, after_connectome, base_state_timeseries, after_state_timeseries) in enumerate(dataloader):
        # print(base_timeseries[0].shape, after_timeseries[0].shape, demo[0].shape, base_connectome[0].shape, after_connectome[0].shape, base_state_timeseries[0].shape, after_state_timeseries[0].shape)
        # torch.Size([456, 53]) torch.Size([456, 53]) torch.Size([5]) torch.Size([1378]) torch.Size([1378]) torch.Size([5, 399]) torch.Size([5, 399])
        
        id1 = base_state_path[0].split('/')[-2]
        id2 = after_state_path[0].split('/')[-2]
        print(base_state_path, after_state_path, id1, id2, paired_session_list[i])
        area_timeseries1 = loadmat(f'./dynamic_volumne/dFNC_dataset/{id1}/area_timeseries.mat')['timeseries']
        area_timeseries2 = loadmat(f'./dynamic_volumne/dFNC_dataset/{id2}/area_timeseries.mat')['timeseries']
        FNC = rest_FNC([area_timeseries2.T, area_timeseries1.T])
        after_connectome = FNC[0][np.triu_indices(53, k=1)]
        base_connectome = FNC[0][np.triu_indices(53, k=1)]


        # demo age_base, predict_duration, gender_base, birth_base, headmotion_base 
        
        if (np.isnan(demo[0][0])) | (demo[0][1]*7 > 90) | (base_state_timeseries[0].shape[1]<200) | (after_state_timeseries[0].shape[1]<200):
            print(demo[0][1]*7 , base_state_timeseries[0].shape[1], after_state_timeseries[0].shape[1])
            continue
        TR_base = 1 if (demo[0][5]-0.72)<0.001 else 0
        TR_after =  1 if (demo[0][6]-0.72)<0.001 else 0
        site = (TR_base+TR_after)/2
        site_list.append(site)
        
        # base_modular_score = calculate_modularity_score(back_upper_triangles(base_connectome[0], 53, k=1))
        # after_modular_score = calculate_modularity_score(back_upper_triangles(after_connectome[0], 53, k=1))
        # base_modular_list.append(base_modular_score )
        # after_modular_list.append(after_modular_score)
        
        base_asymmetry = np.array([calculate_asymmetric_features(timeseries) for timeseries in base_state_timeseries[0].numpy()])#[ 5]
        after_asymmetry = np.array([calculate_asymmetric_features(timeseries) for timeseries in after_state_timeseries[0].numpy()]) #[ 5]
        base_asymmetry_list.append(base_asymmetry)
        after_asymmetry_list.append(after_asymmetry)
        
        demo_list.append(demo[0]) # age_base, predict_duration, gender_base, birth_base, headmotion_base 

        change_list.append((after_connectome-base_connectome).flatten()) #/(demo[0, 1]/7))
        baseline_list.append(base_connectome.flatten())
        after_list.append(after_connectome.flatten())
        session_infor_list.append(paired_session_list[i])
        
    print('total paired sample number:', len(demo_list))
    baseline_list = np.array(baseline_list)
    after_list = np.array(after_list)
    change_list = np.array(change_list)
    site_list = np.array(site_list)
    demo_list = np.array(demo_list)
    session_infor_list = np.array(session_infor_list)

    # base_modular_list = np.array(base_modular_list)
    base_asymmetry_list = np.array(base_asymmetry_list)
    # after_modular_list = np.array(after_modular_list)
    after_asymmetry_list = np.array(after_asymmetry_list)


    string = ''
    print(len(site_list), len(change_list))
    np.save(f'change_list_{state_number}{string}.npy', change_list)
    np.save(f'baseline_list_{state_number}{string}.npy', baseline_list)
    np.save(f'after_list_{state_number}{string}.npy', after_list)
    np.save(f'demo_list_{state_number}{string}.npy', demo_list)
    np.save(f'site_list_{state_number}{string}.npy', site_list)
    np.save(f'session_list_{state_number}{string}.npy', session_infor_list)
    
    # np.save(f'baseline_modular_list_{state_number}{string}.npy', base_modular_list)
    # np.save(f'after_modular_list_{state_number}{string}.npy', after_modular_list)
    np.save(f'baseline_FA_list_{state_number}{string}_duration.npy', base_asymmetry_list)
    np.save(f'after_FA_list_{state_number}{string}_duration.npy', after_asymmetry_list)

    
    return None


def calculate_asymmetric_features(signal):
    
    """
    Calculate Skewness, Peak-to-Trough Ratio, and Duration Asymmetry.
    
    Parameters:
        signal (array): Input time series.
        sampling_rate (float): Sampling rate of the signal.
    
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


def calculate_modularity_score(adj_matrix):
    import community
    import networkx as nx
    if adj_matrix[0, 0] !=1:
        for i in range(len(adj_matrix)):
            adj_matrix[i,i] == 1
    # adj_matrix = np.abs(adj_matrix)
    adj_matrix[adj_matrix<0]=0
    n_nodes = adj_matrix.shape[0]
    # Create a graph with edges above the threshold
    G = nx.Graph()
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            weight = adj_matrix[i, j]
            G.add_edge(i, j, weight=weight)

    # Louvain community detection
    partition = community.best_partition(G, weight='weight')
    
    # Compute modularity
    modularity_score = community.modularity(partition, G, weight='weight')


    return modularity_score 


def collect_dFNC_data():
    path_list, TR_list, session_list, number_list = dataset_info(filter=False)
    timeseries_list = load_ICA_data(session_list, number_list)
    
    age_list = session_infomation_whole( session_list, 'corrected_age_week')*7
    dFNC_list = dynamic_corr(timeseries_list, TR_list, slide_window=45)
    dFNC_list_total = []
    dFNC_list_one = []
    dFNC_list_two = []
    session_name_list = []
    for session, dFNC, number, age in zip( session_list, dFNC_list, number_list, age_list ):
        print(session, number)
        dFNC = np.array([matrix[np.triu_indices(matrix.shape[0], k=1)] for matrix in dFNC]).T
        dFNC = dFNC - dFNC.mean(axis=1, keepdims=True) # [uptriangle, timepoint]
        if not os.path.exists(f'./dynamic_volumne/dFNC_dataset/{session}_{number}'):
            os.makedirs(f'./dynamic_volumne/dFNC_dataset/{session}_{number}')
        savemat(f'./dynamic_volumne/dFNC_dataset/{session}_{number}/dFNC_{session}_{number}.mat', {'dFNC': dFNC})
        dFNC_list_total.append(dFNC)
        session_name_list.append(f'{session}_{number}')
        
    np.savetxt('unpaired_session_name.txt', session_name_list, fmt='%s')
    
    return None


def unpaired_state_multifeatures(state_number=6, string=''):

    state_path = f'./dynamic_volumne/ICA_results/component_{state_number}_pattern.mat'
    state_component = loadmat(state_path)['components']
    state_component = state_component/state_component.std(axis=1, keepdims=True) # [6, 1378]
    inv_state_component = np.linalg.pinv(state_component).T  # [ 5, uptriangle]

    path_list, TR_list, session_list, number_list = dataset_info()
    timeseries_list = load_ICA_data(session_list, number_list)
    path_list2 = []
    # savedir_list2 = []
    for timeseries, session, number in zip(timeseries_list, session_list, number_list):
        print('save', session, number, timeseries.shape)
        path = f'./dynamic_volumne/dFNC_dataset/{session}_{number}/area_timeseries.mat'
        savemat(path, {'timeseries': timeseries.T})

        path_list2.append(path)
        # savedir_list2.append(savedir )
    np.savetxt('./dynamic_volumne/script/area_timeseries_path.txt', path_list2, fmt='%s')
    # np.savetxt('./dynamic_volumne/script/axe_timeseries_savepath.txt', savedir_list2, fmt='%s')

    rest_list = rest_FNC(timeseries_list) # right
    dFNC_list = dynamic_corr(timeseries_list, TR_list, slide_window=45)
    
    gender =  session_infomation_whole( session_list, 'gender')
    TR =  session_infomation_whole( session_list, 'TR')
    birth = session_infomation_whole( session_list, 'gestational_birth')
    age = session_infomation_whole( session_list, 'corrected_age_week')
    risk = session_infomation_whole( session_list, 'risk') # TP 100% belong to Low Risk
    label = session_infomation_whole( session_list, 'label')
    
    headmotion = session_infomation_whole( session_list, 'head_motion', path_list=path_list)
    indice = label==0
    path_list, TR_list, session_list, number_list = path_list[indice], TR_list[indice], session_list[indice], number_list[indice]
    demo_list = np.array([gender[indice], birth[indice], age[indice], risk[indice], label[indice], TR[indice], headmotion[indice]]).T
    headmotion = headmotion[indice]
    print('typical developement demo shape', demo_list.shape)


    sFNC_list_total = []
    demo_list_new = []
    session_list_new = []

    state_duration_list = []
    state_transport_list = []
    state_wave_list = []
    save_info_list =  []
    state_pattern_list = []
    state_asymmetry_list = []
    axe_energy_list = []
    axe_energy_norm_list = []
  
    for session, demo, dFNC, sFNC, number, path, timeseries in zip(session_list, demo_list, dFNC_list, rest_list, number_list, path_list, timeseries_list ):
        if np.isnan(demo[2]):
            print(demo)
            continue
        TR = demo[5]
        hd = demo[6]
        # print(session, number)
        save_info_list.append([path, float(TR), float(hd)])
        sFNC_tmp = sFNC[np.triu_indices(sFNC.shape[0], k=1)]
        sFNC_list_total.append(sFNC_tmp)
        # modular_score = calculate_modularity_score(sFNC)
        # pattern-unchange state timeseries
        dFNC = np.array([matrix[np.triu_indices(matrix.shape[0], k=1)] for matrix in dFNC]).T
        dFNC = dFNC - dFNC.mean(axis=1, keepdims=True)

        state_dir = f'./dynamic_volumne/dFNC_dataset/{session}_{number}'
        state_timeseries = loadmat(f'{state_dir}/constrained_dFNC_pos_timecourse_5.mat')['timecourse'].T # [ state, timeseries]
      
        state_pattern = loadmat(f'{state_dir}/constrained_dFNC_pos_traj_patterns_5.mat')['components'] # [state, uptriangle]
        state_pattern_list.append(state_pattern)

        asymmetry = np.array([calculate_asymmetric_features(timeseries) for timeseries in state_timeseries]) #[ 5]
        state_asymmetry_list.append(asymmetry)

        demo_list_new.append(demo)
        session_list_new.append(session)

    demo_list_new = np.array(demo_list_new)
    session_list_new = np.array(session_list_new)
    save_info_list =  np.array(save_info_list)

    state_pattern_list = np.array(state_pattern_list)
    state_asymmetry_list = np.array(state_asymmetry_list)
    sFNC_list_total = np.array(sFNC_list_total)

    print(demo_list_new.shape, state_asymmetry_list.shape)

    np.save(f'unpaired_sFNC_list_{state_number}{string}.npy', sFNC_list_total)
    np.save(f'unpaired_demo_list_{state_number}{string}.npy', demo_list_new)
    np.save(f'unpaired_session_list_{state_number}{string}.npy', session_list_new)

    np.save(f'unpaired_path_TR_HD_{state_number}{string}.npy', save_info_list)
    np.save(f'unpaired_state_pattern_{state_number}{string}.npy', state_pattern_list)
    np.save(f'unpaired_state_asymetry_{state_number}{string}.npy', state_asymmetry_list)

    return None


def components_selected_based_on_ICASSO(ICASSO_resutls, save_dir, iq_thr=0.8 ):

    component = ICASSO_resutls['Space']
    print(component.shape)
    Iq =  ICASSO_resutls['Iq'][:,0]
    print(Iq)
    indice = Iq>iq_thr
    component_number = indice.sum()
    print('filtered compnent number ', component_number)
    component =  component[indice]
    print('ICASSO shape:', component.shape)
    print('component mean std', component.mean(axis=1), component.std(axis=1))
    component = component/component.std(axis=1, keepdims=True)
    savemat(f'{save_dir}/component_{component_number}_pattern_pos.mat', {'components': component})
    component = back_upper_triangles(component, 53, k=1)
    savemat(f'{save_dir}/component_{component_number}_pattern_pos_sys.mat', {'components': component})

    plot_state_pattern(component_number, trans=False, vmin='None', vmax='None')

    return None


if __name__ == '__main__':
    print('hello')

    unpaired_state_multifeatures(state_number = 5)
    paired_state_multifeatures(state_number = 5)

