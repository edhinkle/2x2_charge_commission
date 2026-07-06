#!/usr/bin/env bash

#SBATCH --account=dune
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --time=0:06:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=27

DATADIR1="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/ColdOperations/data/2025_Operations_Cold/Leakage_Current_LRS_all_off/"
DATASET1_NAME="Leakage_Current_Cold_Cryo_Pump_Off_LRS_All_Off"
DATE1="2025_09_25"

module load python
srun ./make_2x2_charge_commission_channel_dicts_slurm_task.py ${DATADIR1} ${DATASET1_NAME} ${DATE1}


