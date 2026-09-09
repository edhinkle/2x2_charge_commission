#!/usr/bin/env bash

#SBATCH --account=dune
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=27

DATADIR1="/global/cfs/cdirs/dune/www/data/2x2/nearline_run3/packet/ColdCommissioning/Pedestal_prc1"
DATASET1_NAME="Run3_ColdComissioning_WholeDetectorPedestal-prc1"
DATE1="2026_09_08"

module load python
srun ./make_2x2_charge_commission_channel_dicts_slurm_task.py ${DATADIR1} ${DATASET1_NAME} ${DATE1}


