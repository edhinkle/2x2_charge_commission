#!/usr/bin/env bash

NOM_DSET_IDX=0 # Nominal dataset is index 0 unless otherwise noted
DATADIR1="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/Nominal_Pedestal_Correct_Triggering/"
#DATADIR1="/global/cfs/cdirs/dune/www/data/2x2/nearline/packet/commission/June2024/thresholding_06_07/"
DATASET1_NAME="Nominal_Warm"
DATE1="2025_08_26"
DICT1_NAME="channel_dicts/26August2025_Nominal_Pedestal_Periodic_Reset_Enabled_Mod0123_2025_08_26_FULL_channel_dict.json"

DATADIR2="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/WholeDetector_20250824_1/"
DATASET2_NAME="Periodic_Reset_Disabled_Warm"
DATE2="2025_08_24"
DICT2_NAME="channel_dicts/24August2025_Nominal_Pedestal_Mod0123_2025_08_24_FULL_channel_dict.json"

DATADIR3="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/Nominal_Pedestal_256us/"
DATASET3_NAME="Periodic_Reset_Frequency_256us_Warm"
DATE3="2025_08_26"
DICT3_NAME="channel_dicts/26August2025_Pedestal_Mod0123_256us_Reset_2025_08_26_FULL_channel_dict.json"

DATADIR4="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/Nominal_Pedestal_2560us/"
DATASET4_NAME="Periodic_Reset_Frequency_2560us_Warm"
DATE4="2025_08_26"
DICT4_NAME="channel_dicts/26August2025_Pedestal_Mod0123_2560us_Reset_2025_08_26_FULL_channel_dict.json"

DATADIR5="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/Nominal_Pedestal_25600us/"
DATASET5_NAME="Periodic_Reset_Frequency_25600us_Warm"
DATE5="2025_08_27"
DICT5_NAME="channel_dicts/27August2025_Pedestal_Mod0123_25600us_Reset_2025_08_27_FULL_channel_dict.json"

DATADIR6="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/differential_LRS_1/"
DATASET6_NAME="LRS_On_Not_Taking_Data_Warm"
DATE6="2025_08_27"
DICT6_NAME="channel_dicts/27August2025_Pedestal_Mod0123_LRS_On_Not_Taking_Data_2025_08_27_FULL_channel_dict.json"

YAML_MOD0='/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/2x2_charge_commission/August2025/geometry/multi_tile_layout-2.3.16_mod0_swap_T8T4T7.yaml'
YAML_MOD1='/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/2x2_charge_commission/August2025/geometry/multi_tile_layout-2.3.16_mod1_noswap.yaml'
YAML_MOD2='/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/2x2_charge_commission/August2025/geometry/multi_tile_layout-2.5.16_mod2_swap_T7T8.yaml'
YAML_MOD3='/global/cfs/cdirs/dune/users/ehinkle/nd_prototypes_ana/2x2_charge_commission/August2025/geometry/multi_tile_layout-2.3.16_mod3_swap_T5T8_T9T10.yaml'
MAX_MEAN=50
MAX_STD=5


module load python

#if [ ! -e "${DICT1_NAME}" ]; then
#    echo "${DICT1_NAME} does not exist."
#    python August2025_Charge_Commission_make_channel_dictionary.py -dir ${DATADIR1} -n ${DATASET1_NAME} -d ${DATE1}
#fi
#
#if [ ! -e "${DICT2_NAME}" ]; then
#    echo "${DICT2_NAME} does not exist."
#    python August2025_Charge_Commission_make_channel_dictionary.py -dir ${DATADIR2} -n ${DATASET2_NAME} -d ${DATE2}
#fi

python August2025_Charge_Commission_make_pedestal_plots.py -cd ${DICT1_NAME} ${DICT2_NAME} ${DICT3_NAME} ${DICT4_NAME} ${DICT5_NAME} ${DICT6_NAME} \
                                                           -n ${DATASET1_NAME} ${DATASET2_NAME} ${DATASET3_NAME} ${DATASET4_NAME} ${DATASET5_NAME} ${DATASET6_NAME} \
                                                           -d ${DATE1} ${DATE2} ${DATE3} ${DATE4} ${DATE5} ${DATE6} \
                                                           -idx ${NOM_DSET_IDX} \
                                                           -l ${YAML_MOD0} ${YAML_MOD1} ${YAML_MOD2} ${YAML_MOD3} \
                                                           -mo 0 1 2 3 \
                                                           -mm ${MAX_MEAN} \
                                                           -ms ${MAX_STD}

# SINGLE DATASET
#python August2025_Charge_Commission_make_pedestal_plots.py -cd ${DICT1_NAME} -n ${DATASET1_NAME} -d ${DATE1} -idx ${NOM_DSET_IDX} -l ${YAML_MOD0} ${YAML_MOD1} ${YAML_MOD2} ${YAML_MOD3} -mo 0 1 2 3 -mm ${MAX_MEAN} -ms ${MAX_STD}