function run_dfnc_icasso(data_path, ncomps, trials, save_dir, label)
    gift_path = getenv('GIFT_PATH');
    if ~isempty(gift_path)
        addpath(genpath(fullfile(gift_path, 'GroupICAT', 'icatb', 'toolbox', 'icasso122')));
        addpath(genpath(fullfile(gift_path, 'FastICA_25')));
    end
    if exist('icasso', 'file') ~= 2 || exist('fastica', 'file') ~= 2
        error('Set GIFT_PATH to the GIFT root containing GroupICAT and FastICA_25, or add ICASSO and FastICA to the MATLAB path.');
    end

    input_data = load(data_path, 'timeseries');
    [Iq, S, W, T, sR] = icasso(input_data.timeseries, trials, 'lastEig', ncomps, 'vis', 'off');
    if ~isfolder(save_dir)
        mkdir(save_dir);
    end
    filename = sprintf('ICASSO_%s_%d_results.mat', label, ncomps);
    save(fullfile(save_dir, filename), 'Iq', 'T', 'W', 'S', 'sR');
    disp(Iq);
end
