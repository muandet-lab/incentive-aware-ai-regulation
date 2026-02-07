#!/bin/bash

#SBATCH --job-name=str-master
#SBATCH --output=/tmp/job-%j.out

#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu
#SBATCH --time=4320

JOBDATADIR=`ws create work --space "$SLURM_JOB_ID" --duration "2 00:00:00"`
JOBTMPDIR=/tmp/job-"$SLURM_JOB_ID"

# test for the credentials files
srun test -f ~/CISPA-home/.config/enroot/.credentials

srun mkdir "$JOBTMPDIR"

srun mkdir -p "$JOBDATADIR" "$JOBTMPDIR"/models

srun --container-image=projects.cispa.saarland:5005#css/ngc/pytorch:24.01-py3 python3 $HOME/CISPA-scratch/c01ansi/socialBO/exp_influence.py
