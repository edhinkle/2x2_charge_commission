#!/usr/bin/env bash

#SBATCH --account=dune
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=27

DATADIR1="/global/cfs/cdirs/dune/users/edgarmao/trial_pedestal2"
DATASET1_NAME="trial_pedestal2"
DATE1="2026_07_07"

module load python
srun ./make_2x2_charge_commission_channel_dicts_slurm_task.py ${DATADIR1} ${DATASET1_NAME} ${DATE1}


