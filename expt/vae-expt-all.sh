#!/bin/bash
#SBATCH --job-name=vfae
#SBATCH --account=OD-231488
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time 04:00:00
#SBATCH --mem=8GB
#SBATCH --error=/scratch3/bon136/kmac/results/slurm-%A_%a.err
#SBATCH --output=/scratch3/bon136/kmac/results/slurm-%A_%a.out

# vae-expt.py is converted from notebooks/vae-expt.ipynb with
#   jupyter nbconvert vae-expt.ipynb --to python
# theoryclass={VFAE, VFAESI, VFBAESI}
# gamma= {0.1, 0.5, 0.9, 1.0}

#
# Submit runs from a file with each run per line
#
# submit many jobs as an array of jobs
# use e.g. sbatch -a 0-999 vae-expt-all.sh input_file.txt
# where 0-999 are the range of the indices of the jobs

module load pytorch

IFS=$'\n' read -d '' -r -a lines < ${1}

# Submit job
if [ ! -z "$SLURM_ARRAY_TASK_ID" ]
then
    i=$SLURM_ARRAY_TASK_ID
    echo ${lines[i]}
    eval "${lines[i]}"
else
    echo "Error: Missing array index as SLURM_ARRAY_TASK_ID"
fi


