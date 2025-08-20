#!/usr/bin/env bash

NOM_DSET_IDX=0 # Nominal dataset is index 0 unless otherwise noted
DATADIR1="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/NominalPedestal/"
#DATADIR1="/global/cfs/cdirs/dune/www/data/2x2/nearline/packet/commission/June2024/thresholding_06_07/"
DATASET1_NAME="12August2025_Nominal_Pedestal_Mod123"
DATE1="2025_08_12"
DICT1_NAME="channel_dicts/${DATASET1_NAME}_${DATE1}_channel_dict.json"

#DATADIR2=None
#DATASET2_NAME=None
#DATE2="2025_08_12"

YAML_MOD0='/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/2x2_charge_commission/August2025/geometry/multi_tile_layout-2.3.16_mod0_swap_T8T4T7.yaml'
YAML_MOD1='/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/2x2_charge_commission/August2025/geometry/multi_tile_layout-2.3.16_mod1_noswap.yaml'
YAML_MOD2='/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/2x2_charge_commission/August2025/geometry/multi_tile_layout-2.5.16_mod2_swap_T7T8.yaml'
YAML_MOD3='/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/2x2_charge_commission/August2025/geometry/multi_tile_layout-2.3.16_mod3_swap_T5T8_T9T10.yaml'
MAX_MEAN=50
MAX_STD=5


module load python

if [ ! -e "${DICT1_NAME}" ]; then
    echo "${DICT1_NAME} does not exist."
    python August2025_Charge_Commission_make_channel_dictionary.py -dir ${DATADIR1} -n ${DATASET1_NAME} -d ${DATE1}
fi

python August2025_Charge_Commission_make_pedestal_plots.py -cd ${DICT1_NAME} -n ${DATASET1_NAME} -d ${DATE1} -idx ${NOM_DSET_IDX} -l ${YAML_MOD0} ${YAML_MOD1} ${YAML_MOD2} ${YAML_MOD3} -mo 0 1 2 3 -mm ${MAX_MEAN} -ms ${MAX_STD}