#!/usr/bin/env bash

#SBATCH --account=dune
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-47%24   # adjust to match number of files - 1


NOM_DSET_IDX=0 # Nominal dataset is index 0 unless otherwise noted
DATADIR1="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/cooldown/data/2025_cooldown/packet/"
DATASET1_NAME="File_correctedConfig"
DATE1="2025_09_10"
JSON_DIR="run_parallel/channel_dicts/"
#DICT1_NAME="run_parallel/channel_dicts/File_correctedConfig_2025_09_03_15_39_06_CDT_BATCH0.json"

YAML_MOD0='/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/2x2_charge_commission/August2025/geometry/multi_tile_layout-2.3.16_mod0_swap_T8T4T7.yaml'
YAML_MOD1='/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/2x2_charge_commission/August2025/geometry/multi_tile_layout-2.3.16_mod1_noswap.yaml'
YAML_MOD2='/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/2x2_charge_commission/August2025/geometry/multi_tile_layout-2.5.16_mod2_swap_T7T8.yaml'
YAML_MOD3='/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/2x2_charge_commission/August2025/geometry/multi_tile_layout-2.3.16_mod3_swap_T5T8_T9T10.yaml'
MAX_MEAN=70
MAX_STD=5

module load python

FILES=(${JSON_DIR}/File_correctedConfig_${DATE1}_*_CDT_BATCH*.json)
f=${FILES[$SLURM_ARRAY_TASK_ID]}

srun python August2025_Charge_Commission_make_pedestal_plots.py -cd $f -n ${DATASET1_NAME} -d ${DATE1} -idx ${NOM_DSET_IDX} -l ${YAML_MOD0} ${YAML_MOD1} ${YAML_MOD2} ${YAML_MOD3} -mo 0 1 2 3 -mm ${MAX_MEAN} -ms ${MAX_STD}
