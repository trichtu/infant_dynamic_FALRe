function traj_constrain_ICA(trag_path, save_dir, string, method, priors)
    addpath(genpath('cifti-matlab-master'));
    addpath(genpath('gift/GroupICAT/icatb'));


    time_series_data = load(trag_path).dFNC;
    %% time_series_data = load(trag_path).timeseries;
    %% ICA Code....

    % constrained ICA

    priors_data = load(priors).components;
    disp(size(priors_data));
    disp(size(time_series_data));
    [whitesig, dewhiteM] = icatb_calculate_pca(time_series_data, 5);
    [~, W, A, components] = icatb_icaAlgorithm('MOO-ICAR', whitesig', {'ref_data', priors_data});

    timecourse = dewhiteM*A; %timecourses

    save(strcat(save_dir, '/constrained_', string, '_traj_patterns_5.mat' ), 'components');
    save(strcat(save_dir, '/constrained_', string, '_timecourse_5.mat'), 'timecourse');

    disp( size(timecourse));
    disp( size(components));
    

end



