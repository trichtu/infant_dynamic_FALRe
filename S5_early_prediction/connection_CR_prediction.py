from infant_dataset import session_infomation_whole
from sklearn.ensemble import RandomForestRegressor
import statsmodels.stats.multitest as smm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
import numpy as np
from spontaneous_dynamic_decomposition import back_upper_triangles
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from scipy.stats import t as student_t
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression




def calculate_mean_modularity(pattern):
    domain_list = ['SCN', 'AUD', 'SMN', 'VIS', 'CON', 'DMN', 'CER' ]
    domain_position = np.array([[0,5],[5,7],[7, 16],[16, 25], [25, 42], [42, 49], [49, 53]])

    mean_modularity = np.zeros([7,7])
    for i, domain1 in enumerate(domain_list):
        for j, domain2 in enumerate(domain_list):
            tmp = pattern[ domain_position[i][0]:domain_position[i][1], domain_position[j][0]:domain_position[j][1]]
            if i==j:
                mean = tmp[ np.triu_indices(tmp.shape[1], k=1)].mean()
            else:
                mean = tmp.mean()
            mean_modularity[i,j] = mean
    return mean_modularity


def early_prediction():
    state_number = 5

    datapath = '.'
    session_list = np.load(f'{datapath }/session_list_{state_number}.npy')[:, 0]

    print(len(session_list ))

    baseline_FC = np.load(f'{datapath }/baseline_list_5.npy')
    after_FC = np.load(f'{datapath }/after_list_5.npy')

    # mid age
    demo = np.load(f'{datapath }/demo_list_{state_number}.npy')
    base_age_obs = demo[:, 0]*7
    followup_age_obs = (demo[:, 0]+ demo[:,1])*7
    mid_age_obs = (demo[:, 0]+ 0.5*demo[:,1])*7
    mid_age_obs = mid_age_obs.reshape(-1)
    interval = demo[:,1:2]*7


    # base after FA

    FA_before = np.load(f'{datapath}/baseline_FA_list_5_thre_0.npy')
    FA_after = np.load(f'{datapath}/after_FA_list_5_thre_0.npy')

    FA_before= np.concatenate([ FA_before, FA_before**2 ], axis=1)
    FA_after = np.concatenate([ FA_after, FA_after**2], axis=1)

    # change rate
    change_list = np.load(f'{datapath }/change_list_{state_number}.npy')
    print('change list',change_list.shape, FA_before.shape)
    change_rate = change_list/interval
    change_rate  = np.array([ back_upper_triangles(change, 53, k=1 ) for change in change_rate ])

    # filter less than 2 month intervals 
    indice = (interval.flatten())<9*7
    session_list = session_list[indice]
    change_rate = change_rate[indice]
    FA_before,  FA_after = FA_before[indice],  FA_after[indice]
    mid_age_obs = mid_age_obs[indice]
    followup_age_obs = followup_age_obs[indice]
    base_age_obs = base_age_obs[indice]
    baseline_FC = baseline_FC[indice]
    after_FC = after_FC[indice]
    print(base_age_obs.min(), base_age_obs.max(), (followup_age_obs-base_age_obs).mean(), (followup_age_obs-base_age_obs).std())

    unique_sub_list = list(set([session[:8] for session in session_list]))
    unique_sub_list = np.array(unique_sub_list)
    print('leaving', len(unique_sub_list), len(FA_before))
        
    change_list_modular = np.array([ calculate_mean_modularity(change_pattern) for change_pattern in change_rate ])
    change_list_modular = np.array([ change_pattern[np.tril_indices(7, k=0)] for change_pattern in change_list_modular])
    base_FC_modular =  np.array([ calculate_mean_modularity(back_upper_triangles(pattern, 53, k=1)) for pattern in baseline_FC ])
    base_FC_modular_obs = np.array([  pattern[np.tril_indices(7, k=0)] for pattern in base_FC_modular ])
                                
    after_FC_modular =  np.array([ calculate_mean_modularity(back_upper_triangles(pattern, 53, k=1)) for pattern in after_FC ])
    after_FC_modular_obs = np.array([  pattern[np.tril_indices(7, k=0)] for pattern in after_FC_modular ])

    print('input',change_list_modular.shape)


    all_list = []
    all_obs = []
    all_pre = []

    for trail in range(100):
        performance_list = []
        performance_obs = []
        performance_pre = []

        print('trail:',trail)
        for conn in [ 24]: #range(change_list_modular.shape[1]):
            random_number = np.random.randint(1, 300)
            kf = KFold(n_splits=10, shuffle=True, random_state=random_number)
            y_true_all = []
            y_pred_all = []

            r_max = -3
            p=-2
            estimate = 50
 
            y_true_tmp = []
            y_pred_tmp = []

            # Perform 10-fold cross-validation
            for train_idx, test_idx in kf.split(unique_sub_list):
                train_sub_list, test_sub_list = unique_sub_list[train_idx], unique_sub_list[test_idx]
                train_scan_idx = np.array([True if session[:8] in train_sub_list else False for session in session_list ])
                test_scan_idx = np.array([True if session[:8] in test_sub_list else False for session in session_list ])

                X_train, X_test = FA_before[train_scan_idx], FA_before[test_scan_idx]
                y_train, y_test = change_list_modular[ train_scan_idx, conn], change_list_modular[ test_scan_idx, conn]

                # Train model
                model = RandomForestRegressor(n_estimators=estimate, random_state=42)
                model.fit(X_train, y_train)

                # Predict on test fold
                y_pred = model.predict(X_test)
                y_true_tmp.extend(y_test)
                y_pred_tmp.extend(y_pred)

            r, p = pearsonr(y_true_tmp, y_pred_tmp)

            print(conn, r, p)
            performance_list.append([r, p])
            performance_pre.append(y_pred_all)
            performance_obs.append(y_true_all)

        performance_list  = np.array(performance_list )
        performance_obs = np.array(performance_obs)
        performance_pre = np.array(performance_pre)

        all_list.append(performance_list)
        all_obs.append(performance_obs)
        all_pre.append(performance_pre)
    np.save(f'FA_prediction_rp_{trail}_before_2month_1000.npy', all_list)
    np.save(f'FA_prediction_obs_{trail}_before_2month_1000.npy', all_obs)
    np.save(f'FA_prediction_pre_{trail}_before_2month_1000.npy', all_pre)
    return None


def late_prediction():
    state_number = 5

    datapath = '.'
    session_list = np.load(f'{datapath }/session_list_{state_number}.npy')[:, 0]
    print(len(session_list ))

    baseline_FC = np.load(f'{datapath }/baseline_list_5.npy')
    after_FC = np.load(f'{datapath }/after_list_5.npy')

    # mid age
    demo = np.load(f'{datapath }/demo_list_{state_number}.npy')
    base_age_obs = demo[:, 0]*7
    followup_age_obs = (demo[:, 0]+ demo[:,1])*7
    mid_age_obs = (demo[:, 0]+ 0.5*demo[:,1])*7
    mid_age_obs = mid_age_obs.reshape(-1)
    interval = demo[:,1:2]*7
    
    # base after FA
    FA_before = np.load(f'{datapath}/baseline_FA_list_5_thre_0.npy')
    FA_after = np.load(f'{datapath}/after_FA_list_5_thre_0.npy')

    FA_before= np.concatenate([ FA_before, FA_before**2 ], axis=1)
    FA_after = np.concatenate([ FA_after, FA_after**2], axis=1)

    # change rate
    change_list = np.load(f'{datapath }/change_list_{state_number}.npy')
    print('change list',change_list.shape, FA_before.shape)
    change_rate = change_list/interval
    change_rate  = np.array([ back_upper_triangles(change, 53, k=1 ) for change in change_rate ])

    # filter less than 2 month intervals 
    indice = (interval.flatten())<9*7
    session_list = session_list[indice]
    change_rate = change_rate[indice]
    FA_before,  FA_after = FA_before[indice],  FA_after[indice]
    mid_age_obs = mid_age_obs[indice]
    followup_age_obs = followup_age_obs[indice]
    base_age_obs = base_age_obs[indice]
    baseline_FC = baseline_FC[indice]
    after_FC = after_FC[indice]
    print(base_age_obs.min(), base_age_obs.max(), (followup_age_obs-base_age_obs).mean(), (followup_age_obs-base_age_obs).std())

    unique_sub_list = list(set([session[:8] for session in session_list]))
    unique_sub_list = np.array(unique_sub_list)
    print('leaving', len(unique_sub_list), len(FA_before))
        
    change_list_modular = np.array([ calculate_mean_modularity(change_pattern) for change_pattern in change_rate ])
    change_list_modular = np.array([ change_pattern[np.tril_indices(7, k=0)] for change_pattern in change_list_modular])
    base_FC_modular =  np.array([ calculate_mean_modularity(back_upper_triangles(pattern, 53, k=1)) for pattern in baseline_FC ])
    base_FC_modular_obs = np.array([  pattern[np.tril_indices(7, k=0)] for pattern in base_FC_modular ])
                                
    after_FC_modular =  np.array([ calculate_mean_modularity(back_upper_triangles(pattern, 53, k=1)) for pattern in after_FC ])
    after_FC_modular_obs = np.array([  pattern[np.tril_indices(7, k=0)] for pattern in after_FC_modular ])

    print('input',change_list_modular.shape)


    all_list = []
    all_obs = []
    all_pre = []

    for trail in range(100):
        performance_list = []
        performance_obs = []
        performance_pre = []

        print('trail:',trail)
        for conn in [ 24]: #range(change_list_modular.shape[1]):
            random_number = np.random.randint(1, 300)
            kf = KFold(n_splits=10, shuffle=True, random_state=random_number)
            y_true_all = []
            y_pred_all = []
            p=-2
            estimate = 50
 
            y_true_tmp = []
            y_pred_tmp = []

            # Perform 10-fold cross-validation
            for train_idx, test_idx in kf.split(unique_sub_list):
                train_sub_list, test_sub_list = unique_sub_list[train_idx], unique_sub_list[test_idx]
                train_scan_idx = np.array([True if session[:8] in train_sub_list else False for session in session_list ])
                test_scan_idx = np.array([True if session[:8] in test_sub_list else False for session in session_list ])

                X_train, X_test = FA_after[train_scan_idx], FA_after[test_scan_idx]
                y_train, y_test = change_list_modular[ train_scan_idx, conn], change_list_modular[ test_scan_idx, conn]

                # Train model
                model = RandomForestRegressor(n_estimators=estimate, random_state=42)
                model.fit(X_train, y_train)

                # Predict on test fold
                y_pred = model.predict(X_test)
                y_true_tmp.extend(y_test)
                y_pred_tmp.extend(y_pred)

            r, p = pearsonr(y_true_tmp, y_pred_tmp)

            print(conn, r, p)
            performance_list.append([r, p])
            performance_pre.append(y_pred_all)
            performance_obs.append(y_true_all)

        performance_list  = np.array(performance_list )
        performance_obs = np.array(performance_obs)
        performance_pre = np.array(performance_pre)

        all_list.append(performance_list)
        all_obs.append(performance_obs)
        all_pre.append(performance_pre)
    np.save(f'FA_prediction_rp_{trail}_after_2month_1000.npy', all_list)
    np.save(f'FA_prediction_obs_{trail}_after_2month_1000.npy', all_obs)
    np.save(f'FA_prediction_pre_{trail}_after_2month_1000.npy', all_pre)
    return None



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
    for i in range(28):
        r_scores = np.load(f'FA_prediction_rp_999_before_2month_1000.npy')[:,i, 0]
        res = ci_ttest_repeated_kfold_aggregated_r(r_scores, k=10, alpha=0.05, two_sided=False)
        p_list.append(res['p'])
        mean_r_list.append(res['mean_r'])
        CI_r_list.append(res['ci_r'])
        df_list.append(res['df'])
    mean_r_list = np.around(np.array(mean_r_list),3)
    p_list = np.array(p_list)
    CI_r_list = np.array(CI_r_list)

    accepted, p_corrected, _, _ = multipletests(p_list, alpha=0.05, method="fdr_bh")
    p_list = p_corrected
    p_list_str = np.array([ '<0.001' if p<0.001 else str(np.around(p, 3)) for p in p_list])
    CI_r_list = np.array([ '['+str(np.around(CI[0], 3))+', '+ str(np.around(CI[1], 3))+']' for CI in CI_r_list])
    print(mean_r_list.shape, CI_r_list.shape, p_list.shape)
    print(np.array([mean_r_list, CI_r_list, p_list_str]))
    info = pd.DataFrame(np.array([mean_r_list, CI_r_list, df_list, p_list_str ]).T, columns=['mean r','confidence interval (CI)','degree of freedom','FDR-corrected p'])
    info.to_csv('FA_prediction_2month_before.csv')

    return None


