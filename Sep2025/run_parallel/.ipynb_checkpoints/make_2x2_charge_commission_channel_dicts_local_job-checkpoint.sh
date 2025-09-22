#!/usr/bin/env bash

#SBATCH --account=dune
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=25

DATADIR1="/global/homes/c/cmfang/2x2CRS/"
DATASET1_NAME="File_correctedConfig"
DATE1="2025_09_03"

# module load python
# srun ./make_2x2_charge_commission_channel_dicts_slurm_task.py ${DATADIR1} ${DATASET1_NAME} ${DATE1}

python ./make_2x2_charge_commission_channel_dicts_local_task.py ${DATADIR1} ${DATASET1_NAME} ${DATE1}
