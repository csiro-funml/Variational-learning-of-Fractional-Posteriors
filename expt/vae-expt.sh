#!/bin/bash
#SBATCH --job-name=vfae
#SBATCH --account=OD-228587
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time 04:00:00

# vae-expt.py is converted from notebooks/vae-expt.ipynb with
#   jupyter nbconvert vae-expt.ipynb --to python
# run with sbatch --array 1-11 vae-expt.sh

module load pytorch

# turn off display
export DISPLAY=
epochs=500
skiptrain=

case ${SLURM_ARRAY_TASK_ID} in
# Explicit posterior
  1)
    python vae-expt.py --device cuda:0 ${skiptrain} --epochs ${epochs} --theoryclass VFAE --gamma 1.0
    ;;
  2)
    python vae-expt.py --device cuda:0 ${skiptrain} --epochs ${epochs} --theoryclass VFAE --gamma 0.9
    ;;
  3)
    python vae-expt.py --device cuda:0 ${skiptrain} --epochs ${epochs} --theoryclass VFAE --gamma 0.5
    ;;
  4)
    python vae-expt.py --device cuda:0 ${skiptrain} --epochs ${epochs} --theoryclass VFAE --gamma 0.1
    ;;
# Semi-implicit posterior
  5)
    python vae-expt.py --device cuda:0 ${skiptrain} --epochs ${epochs} --theoryclass VFAESI --gamma 1.0
    ;;
  6)
    python vae-expt.py --device cuda:0 ${skiptrain} --epochs ${epochs} --theoryclass VFAESI --gamma 0.9
    ;;
  7)
    python vae-expt.py --device cuda:0 ${skiptrain} --epochs ${epochs} --theoryclass VFAESI --gamma 0.5
    ;;
  8)
    python vae-expt.py --device cuda:0 ${skiptrain} --epochs ${epochs} --theoryclass VFAESI --gamma 0.1
    ;;
# With Bayes posterior
  9)
    python vae-expt.py --device cuda:0 ${skiptrain} --epochs ${epochs} --theoryclass VFBAESI --gamma 0.9
    ;;
  10)
    python vae-expt.py --device cuda:0 ${skiptrain} --epochs ${epochs} --theoryclass VFBAESI --gamma 0.5
    ;;
  11)
    python vae-expt.py --device cuda:0 ${skiptrain} --epochs ${epochs} --theoryclass VFBAESI --gamma 0.1
    ;;
esac
