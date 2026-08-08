import numpy as np
import pandas as pd

def get_behavior_age_data(behave_file='saccAmp_targetsReached_Data_LMsamp_103125_5deg.xlsx', behave_list=['Saccade Amplitude','% Targets Reached']):
    MRI_session = np.load('unpaired_session_list_5.npy')
    MRI_birth = np.load('unpaired_demo_list_5.npy')[:,1]
    MRI_gender =  np.load('unpaired_demo_list_5.npy')[:,0]
    MRI_subID = np.array([ session[:8] for session in MRI_session ])
    #
    behave_info = pd.read_excel(behave_file)
    behave_session = np.array([ sub.replace("-", "_")+str(age) for sub, age in zip(behave_info['SubID'].values, behave_info['CorrAge'].values)])
    
    behave_data = behave_info[behave_list].values
    behave_age = behave_info['CorrAge'].values
    
    behave_unique_session = np.unique(behave_session)

    behave_subID = np.array( [ session[:8].replace("-", "_") for session in behave_unique_session])
    behave_mean_data = np.array([ np.nanmean(behave_data[behave_session==session], axis=0)  for session in behave_unique_session])
    behave_mean_age = np.array([ behave_age[behave_session==session][0]  for session in behave_unique_session])
    
    behave_corrected_age = []
    behave_gender_list = []
    for i, session in enumerate(behave_unique_session):
        subID = session[:8]
#         birth_weeks = MRI_birth[MRI_subID ==subID][0]
#         corrected_age = behave_mean_age[i]-(40-birth_weeks)*7
        corrected_age = behave_mean_age[behave_unique_session==session][0]
        gender = MRI_gender[MRI_subID ==subID][0]
        behave_corrected_age.append(corrected_age)
        behave_gender_list.append(gender)
    behave_corrected_age = np.array(behave_corrected_age)
    behave_gender_list = np.array(behave_gender_list)
    return behave_subID, behave_mean_data, behave_corrected_age, behave_gender_list 



if __name__ == '__main__':


    behave_subID_list, behave_mean_data_list, behave_corrected_age_list, behave_gender_list = get_behavior_age_data(behave_file='saccAmp_targetsReached_Data_LMsamp_103125_5deg.xlsx', behave_list=['Saccade Amplitude','% Targets Reached'])
    behave_subID_list, behave_mean_data_list, behave_corrected_age_list, behave_gender_list = get_behavior_age_data(behave_file='ETData_forLM_081825.xlsx', behave_list=['EyesFix','MouthFix','EMI'])
    print(behave_subID_list, behave_corrected_age_list)