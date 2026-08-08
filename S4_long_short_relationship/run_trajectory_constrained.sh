#!/bin/bash
#SBATCH --job-name=N20        # Job name
#SBATCH --output=output_%j.txt     # Standard output and error log
#SBATCH --error=error_%j.txt       # Error log
#SBATCH --ntasks=1                 # Number of tasks (usually 1)
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem=20G                   # Memory per node
#SBATCH --time=120:00:00            # Time limit hrs:min:sec


priors=./dynamic_volumne/ICA_results/component_5_pattern_pos.mat
cifti_data=./dynamic_volumne/script/GAM_resutls/CS_derivative_dataset_all.mat
save_dir=./dynamic_volumne/script/GAM_resutls
method=MOO-ICAR

matlab -nodisplay -nosplash -nodesktop -r "traj_constrain_ICA( '${cifti_data}', '${save_dir}', 'Long_short','${method}', '${priors}'); exit;"
