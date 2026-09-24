#!/usr/bin/env bash

#SBATCH --account=dune
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=27

DATADIR1="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/ColdOperations/data/2025_Operations_Cold/Nominal_Configurations"
DATASET1_NAME="Run2_ColdComissioning_NominalPedestal_PumpOff"
DATE1="2026_09_22"

module load python
srun ./make_2x2_charge_commission_channel_dicts_slurm_task.py ${DATADIR1} ${DATASET1_NAME} ${DATE1}


