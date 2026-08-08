
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import surface
from scipy.io import loadmat
from datetime import datetime
from matplotlib.patches import Rectangle

Autism_baby_dir ='/Autism_baby'
workdir = '/infant/dynamic_volumne'


def dataset_info(select_session='None', filter=True):
    info = loadmat('ASD_Baby_ica_parameter_info.mat')['sesInfo']
    file = info['inputFiles'][0,0][0]
    path_list = []
    num_list = []
    for i in range(len(file)):
        path, num = file[i][0][0].split(',')
        # print(i , path, num)
        path_list.append(path)
        num_list.append(num)
    path_list  = np.array(path_list )
    TR_list = np.array(info['TR'][0, 0].flatten())
    session_list = []
    for path in path_list:
        family_ID = path.split('/')[-4]
        children_ID, Session_ID = path.split('/')[-3].split('_')[:2]
        session_list.append(f'{family_ID}_{children_ID}_{Session_ID}')

    session_list = np.array(session_list)
    number_list = np.arange(len(session_list))

    if not isinstance(select_session, str):
        indice_list = match_session(session_list, select_session, type='random')
        indice = np.array(indice_list).flatten()
        path_list, TR_list, session_list, number_list = path_list[indice], TR_list[indice], session_list[indice], number_list[indice]

    print('origin:',  len(session_list), len(path_list))
    
    if filter:
        age_list = session_infomation_whole(session_list, 'corrected_age_week', path_list=path_list)
        
        indice = (age_list< 27) & (age_list != -1)

        path_list, TR_list, session_list, number_list = path_list[indice], TR_list[indice], session_list[indice], number_list[indice]

        headmotion_remove_path_list = ['']
        indice = np.array([path not in headmotion_remove_path_list for path in path_list])
        print('remove high hd scans', (~indice).sum(), len(headmotion_remove_path_list))
        path_list, TR_list, session_list, number_list = path_list[indice], TR_list[indice], session_list[indice], number_list[indice]
        print('dataset samples',len(path_list))

    return path_list, TR_list, session_list, number_list


def session_infomation_whole(session_list,  info, path_list='None'):

    df = pd.read_csv('496nfant_Scans_with_demographics.csv')
    df_session_list = df['session_id'].values

    if info == 'corrected_age_week':
        term = (df['Age'] - 7 * ( 40 - df['Gestational Age in Weeks at Birth']))/7  #week
    if info == 'corrected_age':
        term = df['Age'] - 7 * ( 40 - df['Gestational Age in Weeks at Birth'])  # days
    if info == 'gestational_birth':
        term = df['Gestational Age in Weeks at Birth']
    if info == 'gender':
        term = df['Sex']
    if info == 'evaluation_date':
        term = df['Date Of Evaluation']
    if info == 'label':
        term = df['Diagnosis']
    if info == 'head_motion':
        term = df['headMotion']
    if info == 'TR':
        term = df['TR']
    if info == 'risk':
        term = df['Status']
    if info == 'timepoint':
        term = df['TP']
    if info == 'study':
        term = df['Study']
    if info == 'DV_dataset':
        term = df['DV_dataset']
    if info == 'site':
        term = df['TR']
    new_list = []
    for i, session in enumerate(session_list):
        if session in list(df_session_list):
            if info != 'head_motion':
                vv = term[df_session_list==session].values[0]
            else:
                vv = term[(df_session_list==session)&(df['Path']==path_list[i])].values[0]
            if info == 'site':
                vv = 1 if abs(vv-0.72)<0.001 else 0
            if info == 'evaluation_date':
                vv = datetime.strptime(vv, "%m/%d/%Y")
            if info == 'gender':
                vv = 0 if vv=='male' else 1
            if info == 'risk':
                vv = 0 if vv=='LR' else 1
            if info in ['head_motion', 'TR', 'corrected_age_week','gestational_birth', 'corrected_age' ]:
                vv = float(vv)
            if info in ['TP']:
                vv = int(vv)
            if info == 'label':
                if vv == 'TD':
                    vv = 0
                elif vv == 'AUT':
                    vv = 1
                else:
                    vv = -1

            if info == 'study':
                if vv =='Neuroimaging of Infants at High- and Low-Risk for ASD':
                    vv = 1
                elif vv=='ACE Center 2017: Project 3 - Neuroimaging':
                    vv = 2
                else:
                    vv = 0

            new_list.append(vv)
        else:
            # print('not found:', session, path_list[i])
            new_list.append(-1)
    new_list = np.array(new_list)
    return new_list


def load_ICA_data(session_list, number_list):
    print('load dataset with', len(session_list) , 'sessions')

    component_priors = pd.read_excel('Functional/MatchTable_High_NetworkLabling_Finalized.xlsx')
    component_number = (component_priors['\'GSP_IC_ID\''].values[:53]-1).astype('int') # python begin at 0

    dataset = []
    for session, number in zip(session_list, number_list):
        # (timepoint, 100 components)
        timeseries = nib.load(f'{workdir}/cleaned_ASD_Baby_sub{number+1:03}_timecourses_ica_s1_.nii').get_fdata()
        timeseries_select = np.array(timeseries[:, component_number])
        
        # print(timeseries_select.shape)
        dataset.append(timeseries_select)
    
    return dataset


def rest_FNC(timeseries_list):
    FNC_list = []
    for timeseries in timeseries_list:
        corr = np.corrcoef(timeseries.T)
        corr = fisher_z_transform(corr)
        FNC_list.append(corr)
    FNC_list = np.array(FNC_list)
    return FNC_list


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


def match_session(total_session_list, sample_session_list, type='random'):
    indice_list = []

    for session in sample_session_list:
        indices = [i for i, x in enumerate(total_session_list) if x == session]
        if type == 'random':
            np.random.shuffle(indices)
            indice_list.append(indices[0:1])

        elif type =='all':
            indice_list.append(indices)
        elif type == 'first':
            indice_list.append(indices[0:1])

    return indice_list


def separate_subject(session_list, family_independent=True):
    np.random.shuffle(session_list)
    print('random session list:', session_list[:5])
    sub_id_list = []
    family_list = []
    sample_session_list = []
    for session in session_list:
        family, sub_id, subsession = session[:5], session[:8], session[-2:]
        if sub_id in sub_id_list:
            continue
        elif (family_independent) & (family in family_list):
            continue
        else:
            sub_id_list.append(sub_id)
            family_list.append(family)
            sample_session_list.append(session)

    sub_id_list = np.array(sub_id_list)
    family_list = np.array(family_list)
    sample_session_list = np.array(sample_session_list)
    return sub_id_list, family_list, sample_session_list


def baseline_session(session_list):
    sub_id_list = []
    base_session_list = []
    indice_list = []
    for ii, session in enumerate(session_list):
        sub_id, subsession = session[:8], session[-2:]
        if sub_id in sub_id_list:
            continue
        else:
            if subsession =='01':
                if ((sub_id + '_02') in session_list ) or  ((sub_id + '_03') in session_list) :
                    sub_id_list.append(sub_id) 
                    base_session_list.append(session)
                    index = match_session(session_list, [sub_id + '_01'], type='all')[0]
                    indice_list.append(index)
            elif subsession == '02':
                if ((sub_id + '_01') in session_list ):
                    sub_id_list.append(sub_id) 
                    base_session_list.append(sub_id + '_01')
                    index = match_session(session_list, [sub_id + '_01'], type='all')[0]
                    # index = session_list.index(sub_id + '_01')
                    indice_list.append(index)
                elif ((sub_id + '_03') in session_list ):
                    sub_id_list.append(sub_id) 
                    base_session_list.append(session)
                    index = match_session(session_list, [sub_id + '_02'], type='all')[0]
                    indice_list.append(index)

            elif subsession == '03':
                if ((sub_id + '_01') in session_list ):
                    sub_id_list.append(sub_id) 
                    base_session_list.append(sub_id + '_01')
                    index = match_session(session_list, [sub_id + '_01'], type='all')[0]
                    indice_list.append(index)
                elif ((sub_id + '_02') in session_list ):
                    sub_id_list.append(sub_id) 
                    base_session_list.append(sub_id + '_02')
                    index = match_session(session_list, [sub_id + '_02'], type='all')[0]
                    indice_list.append(index)        

    # indice = np.array(indice)
    base_session_list = np.array(base_session_list)
    sub_id_list = np.array(sub_id_list)
    return indice_list, base_session_list, sub_id_list


def sample_sponta_dynamic_dataset(slide_window=45, select_session_list='None', timeseries_only=False, sample_size=100):
    """
    slide_window: n seconds
    if select_session_list is none, sample_size will be used for samplling
    
    timeseries_only: only get_sample_timesereis, do not calculate dynamic FNC
    """
    if isinstance(select_session_list, str):
        path_list, TR_list, session_list, number_list = dataset_info()
        label = session_infomation_whole(session_list, 'label')
        indice = label==0 
        path_list, TR_list, session_list, number_list = path_list[indice], TR_list[indice], session_list[indice], number_list[indice]
        print('total sample:', len(path_list))
        # one trail dataset shuold be # 1) random sample 100 subject #2) subject-independent
        sub_id_list, family_list, sample_session_list = separate_subject(session_list, family_independent=True)
        if sample_size != 'all' :
            sample_session_list = sample_session_list[ : sample_size]
    else:
        sample_session_list = select_session_list
        path_list, TR_list, session_list, number_list = dataset_info(select_session = select_session_list, filter=False)

    # has problem, one subject has multiple data with the same session number for PA- and AP-encoded scans, so match one of them by random or average
    indice = match_session(session_list, sample_session_list, type = 'random')
    indice = np.array(indice).flatten()
    print('sampling number:', len(sample_session_list), sample_session_list[:5])
    sample_path_list, sample_TR_list, sample_number_list = path_list[indice], TR_list[indice], number_list[indice]

    # [subject, timepoint, 53 components]
    sample_timeseries_list = load_ICA_data(sample_session_list, sample_number_list)
    if timeseries_only:
        return sample_session_list, sample_timeseries_list
    
    # subject [timepoint (different numbers), 53, 53]
    sample_dFNC = dynamic_corr(sample_timeseries_list, sample_TR_list, slide_window=slide_window)
    
    # sample_dFNC = np.concatenate(sample_dFNC, axis=0)
    print('whole dynamic shape:', len(sample_dFNC))

    return sample_session_list, sample_timeseries_list, sample_dFNC



def longitudinal_dataset(FNC=True, first_date_as_baseline=False):
    path_list, TR_list, session_list, number_list = dataset_info()
    session_list = np.array(list(set(session_list)))
    scandate_list = session_infomation_whole(session_list, 'evaluation_date')

    age_list = session_infomation_whole( session_list, 'corrected_age_week' )  

    _, base_session_list, base_sub_id_list = baseline_session(session_list)

    # print('Baseline number:', len(indice), len(base_session_list), len(base_sub_id_list), base_session_list[:5])
    # paired session
    
    paired_longitudinal_session = []
    for session in session_list:
        sub_id, subsession = session[:8], session[-2:]
        if (sub_id in base_sub_id_list) & (session not in base_session_list):
            base_session = base_session_list[base_sub_id_list==sub_id][0].astype('str')

            # match the first scan information in the session, because all age, date is the same within the same session
            index1, index2 = list(session_list).index(base_session), list(session_list).index(session)

            age_base, age_after = age_list[index1], age_list[index2]
            scandate_base, scandate_after = scandate_list[index1], scandate_list[index2]
            # print([base_session, session, age_after-age_base, (scandate_after - scandate_base).days])
            paired_longitudinal_session.append([base_session, session, age_after-age_base, int((scandate_after - scandate_base).days) ])

    if not first_date_as_baseline:
        new_session_list = []
        for session in session_list:
            if session not in base_session_list:
                new_session_list.append(session)
        new_session_list = np.array(new_session_list)
        _, new_base_session_list, new_base_sub_id_list = baseline_session(new_session_list)
        # print('added Baseline number:', len(new_indice), len(new_base_session_list), len(new_base_sub_id_list), new_base_session_list[:5])

        for session in new_session_list:
            sub_id, subsession = session[:8], session[-2:]
            if (sub_id in new_base_sub_id_list) & (session not in new_base_session_list):
                base_session = new_base_session_list[new_base_sub_id_list==sub_id][0].astype('str')

                # match the first scan information in the same session, because all age, date is the same within the same session
                index1, index2 = list(session_list).index(base_session), list(session_list).index(session)

                age_base, age_after = age_list[index1], age_list[index2]
                scandate_base, scandate_after = scandate_list[index1], scandate_list[index2]
                # print([base_session, session, age_after-age_base, (scandate_after - scandate_base).days])
                paired_longitudinal_session.append([base_session, session, age_after-age_base, int((scandate_after - scandate_base).days)])

    paired_longitudinal_session = np.array(paired_longitudinal_session)


    if FNC:
        paired_FNC_list = []
        for base_session, after_session, age_duration, date_days in paired_longitudinal_session:
            indice = match_session(session_list, [base_session, after_session], type='all')
            # baseline
            print('baseline FNC:', base_session)
            timeseries_list = load_ICA_data(session_list[indice[0]], number_list[indice[0]])
            base_FNC = rest_FNC(timeseries_list).mean(axis=0)
            
            # after 
            print('afterline FNC:',after_session)
            timeseries_list = load_ICA_data(session_list[indice[1]], number_list[indice[1]])
            after_FNC = rest_FNC(timeseries_list).mean(axis=0)
           
            # long_diff = after_FNC-base_FNC
            paired_FNC_list.append([base_FNC, after_FNC])

        paired_FNC_list  = np.array(paired_FNC_list )
        print('total paired longitudinal sample:', len(paired_FNC_list), 'paired_FNC_diff:', paired_FNC_list.shape)
        return paired_longitudinal_session, paired_FNC_list
    
    print('total paired longitudinal session:', len(paired_longitudinal_session))
    return paired_longitudinal_session



def longitudinal_trajectory_dataset(need_data=True):
    path_list, TR_list, session_list, number_list = dataset_info()

    age_list = session_infomation_whole(session_list, 'corrected_age_week')  
    missing_index = np.isnan(age_list)
    age_list = age_list[~missing_index]
    path_list, TR_list, session_list, number_list = path_list[~missing_index], TR_list[~missing_index], session_list[~missing_index], number_list[~missing_index]

    indice, base_session_list, base_sub_id_list = baseline_session(session_list)
    print('Baseline number:', len(indice), len(base_session_list), len(base_sub_id_list), base_session_list[:5])
    # paired session
    session_base_sub_id_list = np.array([session[:8] for session in session_list])

    visit_list = np.array([session[-2:] for session in session_list])
    paired_longitudinal_session = []
    for sub_id in  base_sub_id_list: 
        subsession_list = []
        subsession_flat_list = session_list[session_base_sub_id_list == sub_id]
        subage_list = age_list[session_base_sub_id_list == sub_id]
        subnumber_list = number_list[session_base_sub_id_list == sub_id]
        subvisit = visit_list[session_base_sub_id_list == sub_id]
        
        for visit in ['01','02','03']:
            if visit in subvisit:
                subsession_list.append([list(subsession_flat_list[subvisit==visit]), list(subnumber_list[subvisit==visit]), list(subage_list[subvisit==visit])])

        paired_longitudinal_session.append(subsession_list)
        print(subsession_list)

    if need_data:
        paired_FNC_list = []
        for subsession_list in paired_longitudinal_session:
            sub_FNC = []
            for visit in subsession_list:
                
                session, session_number, age = visit[0], visit[1], visit[2]
                timeseries_list = load_ICA_data( session, session_number)
                # FNC = rest_FNC(timeseries_list)
                sub_FNC.append(timeseries_list )
                
            # long_diff = after_FNC-base_FNC
            paired_FNC_list.append(sub_FNC)
        
        print('total paired longitudinal subject:', len(paired_FNC_list))
        return paired_longitudinal_session, paired_FNC_list
    
    print('total paired longitudinal subject:', len(paired_FNC_list))
    return paired_longitudinal_session



def plot_FNC(matrix, savepath='None', upper_matrix=True, vmin='None', vmax='None',  fill_value=1):
    import seaborn as sns
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    import pandas as pd
    component_priors = pd.read_excel('MatchTable_High_NetworkLabling_Finalized.xlsx')
    labels = component_priors['Domain (with hippocampus)'].values[:53] # python begin at 0

    # Generate a random 53x53 matrix
    if upper_matrix:
        matrix = matrix.T+matrix
        np.fill_diagonal(matrix, fill_value)


    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8))
    if (vmin !='None') | ( vmax != 'None') :
        ax = sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap='bwr', vmin=vmin, vmax=vmax, annot=False) #  RdBu
    else:
        ax = sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap='bwr', annot=False)
    
    domain_labelset = ['SCN', 'AUD', 'SMN', 'VIS', 'CON', 'DMN', 'CER']
    domain_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan']
    
    y_position = [ 2.5,  6, 11,  20.5,  33.5,  45.5,  51 ]
    plt.yticks(ticks=y_position, labels=domain_labelset,  ha='center', rotation=90, size=12 )
    x_position = np.array([ 2.5,  6, 11,  20.5,  33.5,  45.5,  51 ])-1
    plt.xticks(ticks=x_position+1, labels=domain_labelset,  ha='center',  rotation=0, size=12)
    ax.tick_params(axis='both', which='both', length=3, color='white')
    
    # draw grid line
    for i in range(1, 53):  # Add horizontal lines
        ax.axhline(i, color='white', linestyle='-', linewidth=0.5)
    
    for j in range(1, 53):  # Add vertical lines
        ax.axvline(j, color='white', linestyle='-', linewidth=0.5)
        
    # draw domain line
    domain_indices = [0, 5, 7, 16, 25, 42, 49, 53]
    for i in domain_indices:  # Add horizontal lines
        ax.axhline(i, color='black', linestyle='-', linewidth=1)
    ax.axhline(53, color='black', linestyle='-', linewidth=1.5)  
    
    for j in domain_indices:  # Add vertical lines
        ax.axvline(j, color='black', linestyle='-', linewidth=1)    
    ax.axvline(53, color='black', linestyle='-', linewidth=1.5)
    
    # plt.show()
    if savepath != 'None':
        plt.savefig(savepath)
    return None


def dataset_information():
    # longitudinal data
    paired_longitudinal_session = longitudinal_dataset(FNC=False, first_date_as_baseline=False)

    subid_list = []
    unique_session_list = []
    for session in paired_longitudinal_session:
        subid = session[:8]
        if subid not in subid_list:
            subid_list.append(subid)
            unique_session_list.append(session)
    
    return None

if __name__ == '__main__':

    path_list, TR_list, session_list, number_list = dataset_info()
    print(len(path_list))
    print(session_list, len(list(set(session_list))))

