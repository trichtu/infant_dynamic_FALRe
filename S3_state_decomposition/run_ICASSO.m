function run_ICASSO(data_path, ncomps, trail, save_dir, string) 

addpath(genpath('cifti-matlab-master'));
addpath(genpath('gift/GroupICAT/icatb'));
addpath(genpath('gift/FastICA_25'));

load(data_path, 'data');
ncomps = str2num(ncomps);
trail = str2num(trail);
[Iq,space,W,time,sR] = icasso(data, trail, 'lastEig', ncomps);
save(strcat(save_dir, '/ICASSO_', string, '_',num2str(ncomps), '_results.mat'), 'Iq','space','W','time','sR');
disp(Iq)
end
