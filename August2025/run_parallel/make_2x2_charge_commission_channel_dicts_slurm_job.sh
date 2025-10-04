#!/usr/bin/env bash

#SBATCH --account=dune
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30

DATADIR1="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/differential_PMalloff/"
DATASET1_NAME="02September2025_Pedestal_Mod0123_PM_AllOff"
DATE1="2025_09_02"

module load python
srun ./make_2x2_charge_commission_channel_dicts_slurm_task.py ${DATADIR1} ${DATASET1_NAME} ${DATE1}


