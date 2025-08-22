#!/usr/bin/env bash

#SBATCH --account=dune
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --time=1:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30

DATADIR1="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/NominalPedestal_v2/"
DATASET1_NAME="13August2025_Nominal_Pedestal_Mod123"
DATE1="2025_08_13"

module load python
srun ./make_2x2_charge_commission_channel_dicts_slurm_task.py ${DATADIR1} ${DATASET1_NAME} ${DATE1}


