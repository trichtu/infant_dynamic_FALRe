import numpy as np
import pandas as pd
from infant_dataset import session_infomation_whole
from sklearn.ensemble import RandomForestRegressor
import statsmodels.stats.multitest as smm
from sklearn.model_selection import cross_val_score, KFold
from visual_behavior_dataset import get_behavior_age_data
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def early_behavior_prediction():

    savepath = './Two_month_prediction'
    # MRI subID
    MRI_session = np.load(f'{savepath}/unpaired_session_list_5.npy')
    MRI_age = np.load(f'{savepath}/unpaired_demo_list_5.npy')[:,2]*7
    MRI_FA = np.load(f'{savepath}/unpaired_state_asymetry_5_final.npy')
    MRI_subID = [ session[:8] for session in MRI_session ]

    # target-reached percent
    behave_subID_list, behave_mean_data_list, behave_corrected_age_list, behave_gender_list = get_behavior_age_data(behave_file='saccAmp_targetsReached_Data_LMsamp_103125_5deg.xlsx', behave_list=['% Targets Reached'])
    # eye-looking
    # behave_subID_list, behave_mean_data_list, behave_corrected_age_list, behave_gender_list = get_behavior_age_data(behave_file='ETData_forLM_081825.xlsx', behave_list=['EyesFix'])

    FA_feature_list = []
    behavior_rate_list = []
    subID_list = []
    mid_age_list = []
    base_age_list = []
    followup_age_list = []
    base_behavior_list = []
    followup_behavior_list = []

    for i, subID in enumerate(MRI_subID):

        sub_age = MRI_age[i]
        behavior_age_range = [sub_age ,sub_age+60] 
        indice = behave_subID_list == subID
        sub_behave_data, sub_corrected_age = behave_mean_data_list[indice], behave_corrected_age_list[indice]
        sindice = (sub_corrected_age >= behavior_age_range[0]) & (sub_corrected_age <= behavior_age_range[1])
        if (sindice).sum()< 2:
            continue
        elif (sindice).sum()==2:
            bb, aa = sub_behave_data[sindice], sub_corrected_age[sindice]

            # no matter which age sequence.
            change_rate = (bb[0]-bb[1])/(aa[0]-aa[1])
    #         print(aa, abs(aa[0]-aa[1]))
            mid_age = (aa[0]+aa[1])/2
            FA_feature_list.append(MRI_FA[i])
            mid_age_list.append(mid_age)
            behavior_rate_list.append(change_rate)
            subID_list.append(subID)
            if aa[0]<aa[1]:
                base_age_list.append(aa[0])
                followup_age_list.append(aa[1])
                base_behavior_list.append(bb[0])
                followup_behavior_list.append(bb[1])
            else:
                base_age_list.append(aa[1])
                followup_age_list.append(aa[0])
                base_behavior_list.append(bb[1])
                followup_behavior_list.append(bb[0])            
            
        else:
            bb, aa=  sub_behave_data[sindice], sub_corrected_age[sindice]
            idx = np.argsort(aa)
            bb, aa = bb[idx], aa[idx]
            for kk in range(len(bb)-1):
                interval = abs(aa[kk]-aa[kk+1])
                if interval<90:
                    change_rate = (bb[kk]-bb[kk+1])/(aa[kk]-aa[kk+1])
                    mid_age = (aa[kk]+aa[kk+1])/2
                    mid_age_list.append(mid_age)
                    FA_feature_list.append(MRI_FA[i])
                    behavior_rate_list.append(change_rate)
                    subID_list.append(subID)
                    if aa[kk]<aa[kk+1]:
                        base_age_list.append(aa[kk])
                        followup_age_list.append(aa[kk+1])
                        base_behavior_list.append(bb[kk])
                        followup_behavior_list.append(bb[kk+1])
                    else:
                        base_age_list.append(aa[kk+1])
                        followup_age_list.append(aa[kk])
                        base_behavior_list.append(bb[kk+1])
                        followup_behavior_list.append(bb[kk])  
                    
    FA_feature_list = np.array(FA_feature_list)
    behavior_rate_list = np.array(behavior_rate_list)
    subID_list = np.array(subID_list)   
    mid_age_list = np.array(mid_age_list)
    base_age_list = np.array(base_age_list)
    followup_age_list = np.array(followup_age_list)
    base_behavior_list = np.array(base_behavior_list)
    followup_behavior_list = np.array(followup_behavior_list)

    subID_list = np.array(subID_list)  
    X = np.concatenate([FA_feature_list, FA_feature_list**2], axis=1)
    unique_sub_list =np.unique(subID_list)

    all_list = []
    all_obs = []
    all_pre = []

    for conn in [0]: # range(behavior_rate_list.shape[1]):
        performance_list = []
        performance_obs = []
        performance_pre = []

        
        for trail in range(1000):
            print('trail:',trail)
            random_number = np.random.randint(1, 600)
            kf = KFold(n_splits=10, shuffle=True, random_state=random_number)
            y_true_all = []
            y_pred_all = []

            p=-2
            estimate=50
            # Perform 10-fold cross-validation
            for train_idx, test_idx in kf.split(unique_sub_list):
                train_sub_list, test_sub_list = unique_sub_list[train_idx], unique_sub_list[test_idx]
                train_scan_idx = np.array([True if subID in train_sub_list else False for subID in subID_list ])
                test_scan_idx = np.array([True if subID in test_sub_list else False for subID in subID_list ])
                X_train, X_test = X[train_scan_idx], X[test_scan_idx]
                y_train, y_test = behavior_rate_list[ train_scan_idx, conn], behavior_rate_list[ test_scan_idx, conn]

                # Train model
                model = RandomForestRegressor(n_estimators=estimate, random_state=random_number)
                model.fit(X_train, y_train)

                # Predict on test fold
                y_pred = model.predict(X_test)

                # Collect all predictions and true values
                y_true_all.extend(y_test)
                y_pred_all.extend(y_pred)

            r, p = pearsonr(y_true_all, y_pred_all)

            print(r, p)
            performance_list.append([r, p])
            performance_pre.append(y_pred_all)
            performance_obs.append(y_true_all)
     
        all_list.append(performance_list)
        all_obs.append(performance_obs)
        all_pre.append(performance_pre)

    np.save(f'FA_behavior_rp_{trail}_before_2month_1000_percent.npy', all_list)
    np.save(f'FA_behavior_obs_{trail}_before_2month_1000_percent.npy', all_obs)
    np.save(f'FA_behavior_pre_{trail}_before_2month_1000_percent.npy', all_pre)



def late_behavior_prediction():

    savepath = './Two_month_prediction'
    # MRI subID
    MRI_session = np.load(f'{savepath}/unpaired_session_list_5.npy')
    MRI_age = np.load(f'{savepath}/unpaired_demo_list_5.npy')[:,2]*7
    MRI_FA = np.load(f'{savepath}/unpaired_state_asymetry_5_final.npy')
    MRI_subID = [ session[:8] for session in MRI_session ]


    # target-reached percent
    behave_subID_list, behave_mean_data_list, behave_corrected_age_list, behave_gender_list = get_behavior_age_data(behave_file='saccAmp_targetsReached_Data_LMsamp_103125_5deg.xlsx', behave_list=['% Targets Reached'])
    # eye-looking
    # behave_subID_list, behave_mean_data_list, behave_corrected_age_list, behave_gender_list = get_behavior_age_data(behave_file='ETData_forLM_081825.xlsx', behave_list=['EyesFix'])

    FA_feature_list = []
    behavior_rate_list = []
    subID_list = []
    mid_age_list = []
    base_age_list = []
    followup_age_list = []
    base_behavior_list = []
    followup_behavior_list = []

    for i, subID in enumerate(MRI_subID):

        sub_age = MRI_age[i]
        behavior_age_range = [sub_age-60 , sub_age] 
        indice = behave_subID_list == subID
        sub_behave_data, sub_corrected_age = behave_mean_data_list[indice], behave_corrected_age_list[indice]
        sindice = (sub_corrected_age >= behavior_age_range[0]) & (sub_corrected_age <= behavior_age_range[1])
        if (sindice).sum()< 2:
            continue
        elif (sindice).sum()==2:
            bb, aa = sub_behave_data[sindice], sub_corrected_age[sindice]

            # no matter which age sequence.
            change_rate = (bb[0]-bb[1])/(aa[0]-aa[1])
    #         print(aa, abs(aa[0]-aa[1]))
            mid_age = (aa[0]+aa[1])/2
            FA_feature_list.append(MRI_FA[i])
            mid_age_list.append(mid_age)
            behavior_rate_list.append(change_rate)
            subID_list.append(subID)
            if aa[0]<aa[1]:
                base_age_list.append(aa[0])
                followup_age_list.append(aa[1])
                base_behavior_list.append(bb[0])
                followup_behavior_list.append(bb[1])
            else:
                base_age_list.append(aa[1])
                followup_age_list.append(aa[0])
                base_behavior_list.append(bb[1])
                followup_behavior_list.append(bb[0])            
            
        else:
            bb, aa=  sub_behave_data[sindice], sub_corrected_age[sindice]
            idx = np.argsort(aa)
            bb, aa = bb[idx], aa[idx]
            for kk in range(len(bb)-1):
                interval = abs(aa[kk]-aa[kk+1])
                if interval<90:
                    change_rate = (bb[kk]-bb[kk+1])/(aa[kk]-aa[kk+1])
                    mid_age = (aa[kk]+aa[kk+1])/2
                    mid_age_list.append(mid_age)
                    FA_feature_list.append(MRI_FA[i])
                    behavior_rate_list.append(change_rate)
                    subID_list.append(subID)
                    if aa[kk]<aa[kk+1]:
                        base_age_list.append(aa[kk])
                        followup_age_list.append(aa[kk+1])
                        base_behavior_list.append(bb[kk])
                        followup_behavior_list.append(bb[kk+1])
                    else:
                        base_age_list.append(aa[kk+1])
                        followup_age_list.append(aa[kk])
                        base_behavior_list.append(bb[kk+1])
                        followup_behavior_list.append(bb[kk])  
                    
    FA_feature_list = np.array(FA_feature_list)
    behavior_rate_list = np.array(behavior_rate_list)
    subID_list = np.array(subID_list)   
    mid_age_list = np.array(mid_age_list)
    base_age_list = np.array(base_age_list)
    followup_age_list = np.array(followup_age_list)
    base_behavior_list = np.array(base_behavior_list)
    followup_behavior_list = np.array(followup_behavior_list)

    # begin prediction
    subID_list = np.array(subID_list)  
    X = np.concatenate([FA_feature_list, FA_feature_list**2], axis=1)
    unique_sub_list =np.unique(subID_list)

    all_list = []
    all_obs = []
    all_pre = []

    for conn in [0]: # range(behavior_rate_list.shape[1]):
        performance_list = []
        performance_obs = []
        performance_pre = []

        
        for trail in range(1000):
            print('trail:',trail)
            random_number = np.random.randint(1, 600)
            kf = KFold(n_splits=10, shuffle=True, random_state=random_number)
            y_true_all = []
            y_pred_all = []

            p=-2
            estimate=50
            # Perform 10-fold cross-validation
            for train_idx, test_idx in kf.split(unique_sub_list):
                train_sub_list, test_sub_list = unique_sub_list[train_idx], unique_sub_list[test_idx]
                train_scan_idx = np.array([True if subID in train_sub_list else False for subID in subID_list ])
                test_scan_idx = np.array([True if subID in test_sub_list else False for subID in subID_list ])
                X_train, X_test = X[train_scan_idx], X[test_scan_idx]
                y_train, y_test = behavior_rate_list[ train_scan_idx, conn], behavior_rate_list[ test_scan_idx, conn]

                # Train model
                model = RandomForestRegressor(n_estimators=estimate, random_state=random_number)
                model.fit(X_train, y_train)

                # Predict on test fold
                y_pred = model.predict(X_test)

                # Collect all predictions and true values
                y_true_all.extend(y_test)
                y_pred_all.extend(y_pred)

            r, p = pearsonr(y_true_all, y_pred_all)

            print(r, p)
            performance_list.append([r, p])
            performance_pre.append(y_pred_all)
            performance_obs.append(y_true_all)
     
        all_list.append(performance_list)
        all_obs.append(performance_obs)
        all_pre.append(performance_pre)

    np.save(f'FA_behavior_rp_{trail}_after_2month_1000_percent.npy', all_list)
    np.save(f'FA_behavior_obs_{trail}_after_2month_1000_percent.npy', all_obs)
    np.save(f'FA_behavior_pre_{trail}_after_2month_1000_percent.npy', all_pre)






def _fisher_z(r):
    r = np.asarray(r, dtype=float)
    r = np.clip(r, -0.999999, 0.999999)   # avoid inf
    return np.arctanh(r)

def _inv_fisher_z(z):
    return np.tanh(z)

def ci_ttest_repeated_kfold_aggregated_r(
    r_scores,             # shape (R,) : one r per repeat (from concatenated OOF preds)
    k,                    # folds per repeat, e.g. 5
    alpha=0.05,
    two_sided=False
):
    """
    Corrected CI / t-test for repeated k-fold when each repeat yields ONE aggregated score.
    Uses Bouckaert & Frank's correction: c = 1/R + 1/(k-1).
    Returns dict with mean_r, t, df, p, ci_r=(lo, hi), se_z, R, k, c.
    """
    z = _fisher_z(r_scores).ravel()
    R = z.size
    if R < 2:
        raise ValueError("Need at least 2 repeats.")
    if k <= 1:
        raise ValueError("k must be >= 2.")

    c = 1.0/R + 1.0/(k - 1.0)               # correction for aggregated-per-repeat scores
    mean_z = float(z.mean())
    s2 = float(z.var(ddof=1))
    se = np.sqrt(max(0.0, c * s2))
    df = R - 1
    t_stat = np.inf if se == 0 else mean_z / se

    if two_sided:
        p = 2 * student_t.sf(abs(t_stat), df)
        tcrit = student_t.ppf(1 - alpha/2, df)
    else:
        p = student_t.sf(t_stat, df)        # one-sided H1: mean_r > 0
        tcrit = student_t.ppf(1 - alpha, df)

    ci_z = (mean_z - tcrit*se, mean_z + tcrit*se)
    ci_r = (_inv_fisher_z(ci_z[0]), _inv_fisher_z(ci_z[1]))
    return {
        "mean_r": _inv_fisher_z(mean_z),
        "t": t_stat,
        "df": df,
        "p": p,
        "ci_r": ci_r,
        "se_z": se,
        "R": R,
        "k": k,
        "c": c,
    }


def significance():
    from statsmodels.stats.multitest import multipletests
    from spontaneous_dynamic_decomposition import back_upper_triangles
    import pandas as pd

    mean_r_list = []
    CI_r_list = []
    p_list = []
    df_list = []
    r_scores = np.load(f'FA_behavior_rp_999_before_2month_1000_target.npy')[:,0, 0]
    res = ci_ttest_repeated_kfold_aggregated_r(r_scores, k=10, alpha=0.05, two_sided=False)
    p_list.append(res['p'])
    mean_r_list.append(res['mean_r'])
    CI_r_list.append(res['ci_r'])
    df_list.append(res['df'])

    r_scores = np.load(f'FA_behavior_rp_999_before_2month_1000_looking.npy')[:,0, 0]
    res = ci_ttest_repeated_kfold_aggregated_r(r_scores, k=10, alpha=0.05, two_sided=False)
    p_list.append(res['p'])
    mean_r_list.append(res['mean_r'])
    CI_r_list.append(res['ci_r'])
    df_list.append(res['df'])
    
    accepted, p_corrected, _, _ = multipletests(p_list, alpha=0.05, method="fdr_bh")
    p_list = p_corrected

    info = pd.DataFrame(np.array([mean_r_list, CI_r_list, df_list, p_list ]).T, columns=['mean r','confidence interval (CI)','degree of freedom','FDR-corrected p'])
    info.to_csv('FA_behavior_2month_before.csv')

    return None