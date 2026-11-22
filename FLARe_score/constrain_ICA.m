function constrain_ICA(input_path, save_dir, label, method, priors)
    gift_path = getenv('GIFT_PATH');
    if ~isempty(gift_path)
        toolbox_path = fullfile(gift_path, 'GroupICAT', 'icatb');
        if isfolder(toolbox_path)
            gift_path = toolbox_path;
        end
        addpath(genpath(gift_path));
    end
    if exist('icatb_calculate_pca', 'file') ~= 2 || exist('icatb_icaAlgorithm', 'file') ~= 2
        error('Set GIFT_PATH to the GIFT root or icatb directory, or add GIFT to the MATLAB path.');
    end

    time_series_data = load(input_path, 'timeseries').timeseries;
    priors_data = load(priors, 'components').components;
    
    if size(time_series_data, 1) ~= size(priors_data, 2)
        error('Input rows must match the number of features in the ICA priors.');
    end
    n_components = size(priors_data, 1);
    [whitesig, dewhiteM] = icatb_calculate_pca(time_series_data, n_components);
    [~, ~, A, components] = icatb_icaAlgorithm(method, whitesig', {'ref_data', priors_data});
    timecourse = dewhiteM * A;

    if ~isfolder(save_dir)
        mkdir(save_dir);
    end
    save(fullfile(save_dir, ['constrained_', label, '_traj_patterns_5.mat']), 'components');
    save(fullfile(save_dir, ['constrained_', label, '_timecourse_5.mat']), 'timecourse');
end
