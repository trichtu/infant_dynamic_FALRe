from visual_behavior_dataset import get_behavior_age_data
import nibabel as nib
import pandas as pd
import numpy as np
from GAM_trajectory import GAM_trajectory_estimate
import matplotlib.pyplot as plt

def behavior_trajectory():

    feature_num=1
    behave_subID_list, behave_mean_data_list, behave_corrected_age_list, behave_gender_list = get_behavior_age_data(behave_file='saccAmp_targetsReached_Data_LMsamp_103125_5deg.xlsx', behave_list=['Saccade Amplitude','% Targets Reached'])

    # feature_num=0
    # behave_subID_list, behave_mean_data_list, behave_corrected_age_list, behave_gender_list = get_behavior_age_data(behave_file='ETData_forLM_081825.xlsx', behave_list=['EyesFix','MouthFix','EMI'])
    print(behave_subID_list, behave_corrected_age_list)


    Be_data_new1 = np.array(behave_mean_data_list)
    gender_list_new = np.array( behave_gender_list)
    age_list_new = np.array(behave_corrected_age_list)
    hd_list_new = np.zeros_like(gender_list_new)
    site_new = np.zeros_like(gender_list_new)
    weights = np.ones_like(gender_list_new)

    # print(len(gender_list_new), len(behave_subID_list), len(age_list_new), len(Be_data_new1))
    # print(Be_data_new1, age_list_new, gender_list_new, hd_list_new)

    _, Be_traj1, confidence, derivative_behave_2 = GAM_trajectory_estimate( Be_data_new1[:,feature_num], age_list_new, gender_list_new, hd_list_new, site_new, weights=weights, gradients=True)

    behavior_name = 'Targets_Reached'
    # behavior_name = 'Eye_looking'
    filename = behavior_name+'_traj'
    indice = age_list_new<180
    x = age_list_new[indice]
    y = Be_data_new1[indice, feature_num]
    print(x.shape, y.shape)
    x1= np.arange(0,181)
    y_traj = Be_traj1

    fig, ax = plt.subplots(figsize=(8,4))  # Small figure size for publication

    # Scatter plot
    ax.scatter( x, y, s=50, facecolors='orange', edgecolors='black', linewidth=1)

    # Regression line
    # ax.plot(x1, y_traj, color='black', linewidth=3)
    ax.plot(x1[50:], Be_traj1[50:] , color='orange', linewidth=5)
    ax.fill_between(x1[50:],confidence[50:,0], confidence[50:,1], color="orange", alpha=0.1)
    # Labels
    ax.set_xlabel('Corrected Age (days)', size=15)
    ax.set_ylabel('% Targets Reached',size=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=15,right=False)   # Remove left ticks
    ax.tick_params(labelsize=15, top=False)    # Remove top ticks


    plt.xlim([10, 180])
    plt.ylim([-10, 110])
    # Tight layout
    plt.tight_layout()

    # Save as vector format suitable for Nature
    print(f"./figure/new/{filename}.svg")
    plt.savefig(f"./figure/new/{filename}.svg", dpi=700)  
    plt.show()


def bootstrap_reached_behavior_CRtraj(trail=0):
    print('trail number',trail)

    feature_num=1
    behave_subID_list, behave_mean_data_list, behave_corrected_age_list, behave_gender_list = get_behavior_age_data(behave_file='saccAmp_targetsReached_Data_LMsamp_103125_5deg.xlsx', behave_list=['Saccade Amplitude','% Targets Reached'])

    age_list_new = np.array(behave_corrected_age_list)
    tmp = np.random.randint(0, 10, size=age_list_new.shape)
    thr = np.percentile(tmp, 80)
    indice = tmp<thr
    
    age_list_new = age_list_new[indice]
    Be_data_new1 = np.array(behave_mean_data_list)[indice]
    gender_list_new = np.array( behave_gender_list)[indice]
    hd_list_new = np.zeros_like(behave_gender_list)[indice]
    site_list = np.zeros_like(behave_gender_list)[indice]
    weights = np.ones_like(behave_gender_list)[indice]
    print(len(gender_list_new), len(behave_subID_list), len(age_list_new), len(Be_data_new1))
    # print(Be_data_new1, age_list_new, gender_list_new, hd_list_new)

    _, Be_traj1, confidence, derivative_behave_2 = GAM_trajectory_estimate( Be_data_new1[:,feature_num], age_list_new, gender_list_new, hd_list_new,  site_list, weights=weights, gradients=True)

    np.save(f'./behavior_bootstrap/Reached_CR_traj_{trail}.npy', derivative_behave_2)
    return None


    
def bootstrap_eye_behavior_CRtraj(trail=0):
    print('trail number',trail)

    feature_num=0
    behave_subID_list, behave_mean_data_list, behave_corrected_age_list, behave_gender_list = get_behavior_age_data(behave_file='ETData_forLM_081825.xlsx', behave_list=['EyesFix','MouthFix','EMI'])
    # print(behave_subID_list, behave_corrected_age_list)


    age_list_new = np.array(behave_corrected_age_list)
    tmp = np.random.randint(0, 10, size=age_list_new.shape)
    thr = np.percentile(tmp, 80)
    indice = tmp<thr
    
    age_list_new = age_list_new[indice]
    Be_data_new1 = np.array(behave_mean_data_list)[indice]
    gender_list_new = np.array( behave_gender_list)[indice]
    hd_list_new = np.zeros_like(behave_gender_list)[indice]
    site_list = np.zeros_like(behave_gender_list)[indice]
    weights = np.ones_like(behave_gender_list)[indice]
    print(len(gender_list_new), len(behave_subID_list), len(age_list_new), len(Be_data_new1))

    _, Be_traj1, confidence, derivative_behave_2 = GAM_trajectory_estimate( Be_data_new1[:,feature_num], age_list_new, gender_list_new, hd_list_new, site_list, weights=weights, gradients=True)

    np.save(f'./figure/behavior_bootstrap/Eye_CR_traj_{trail}.npy', derivative_behave_2)

    return None




if __name__ == '__main__':

    behavior_trajectory()