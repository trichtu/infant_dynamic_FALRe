#!/bin/bash
#SBATCH --job-name=gift          # Job name
#SBATCH --output=output_%j.txt     # Standard output and error log
#SBATCH --error=error_%j.txt       # Error log
#SBATCH --ntasks=1                 # Number of tasks (usually 1)
#SBATCH --cpus-per-task=2          # Number of CPU cores per task
#SBATCH --mem=5G                   # Memory per node
#SBATCH --time=04:00:00            # Time limit hrs:min:sec
#SBATCH --partition=qTRD
#SBATCH --array=0-523

source ~/.bashrc

sub_list=./dynamic_volumne/script/unpaired_session_name.txt

sub=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ${sub_list})
cifti_data=./dynamic_volumne/dFNC_dataset/${sub}/dFNC_${sub}.mat
save_dir=./dynamic_volumne/dFNC_dataset/${sub}
priors=./dynamic_volumne/ICA_results/component_5_pattern_pos.mat
method=MOO-ICAR
matlab -nodisplay -nosplash -nodesktop -r "traj_constrain_ICA( '${cifti_data}', '${save_dir}', 'dFNC_pos','${method}', '${priors}'); exit;"




