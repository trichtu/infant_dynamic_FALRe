#!/bin/bash
#SBATCH --job-name=N20        # Job name
#SBATCH --output=output_%j.txt     # Standard output and error log
#SBATCH --error=error_%j.txt       # Error log
#SBATCH --ntasks=1                 # Number of tasks (usually 1)
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem=200G                   # Memory per node
#SBATCH --time=120:00:00            # Time limit hrs:min:sec


source ~/.bashrc
#var name: data
inputdata_path=./dynamic_volumne/ICA_results/all_remean_dFNC.mat

cd ./dynamic_volumne/script

ncomps=10
trail=100
string=one
inputdata_path=./dynamic_volumne/script/one_dFNC.mat
echo ncomps: $ncomps trail times: $trail
savedir=./dynamic_volumne/ICA_results
matlab -nodisplay -nosplash -nodesktop -r "run_ICASSO('${inputdata_path}','${ncomps}', '${trail}', '${savedir}','${string}'); exit;"
echo $string

