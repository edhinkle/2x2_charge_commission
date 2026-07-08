#!/usr/bin/env bash

NOM_DSET_IDX=0 # Nominal dataset is index 0 unless otherwise noted
'''
DATADIR1="/global/cfs/cdirs/dune/users/edgarmao/trial_pedestal/"
#DATADIR1="/global/cfs/cdirs/dune/www/data/2x2/nearline/packet/commission/June2024/thresholding_06_07/"
DATASET1_NAME="trial_pedestal"
DATE1="2026_07_07"
DICT1_NAME="/global/cfs/cdirs/dune/users/edgarmao/2x2_crs_analysis/2x2_charge_commission/July2026/channel_dicts/trial_pedestal_2026_07_07_FULL_channel_dict.json"
'''

DATADIR1="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/Nominal_Pedestal_Correct_Triggering/"
#DATADIR1="/global/cfs/cdirs/dune/www/data/2x2/nearline/packet/commission/June2024/thresholding_06_07/"
DATASET1_NAME="Nominal_Pedestal_Correct_Triggering"
DATE1="2025_08_26"
DICT1_NAME="/global/cfs/cdirs/dune/users/edgarmao/2x2_crs_analysis/2x2_charge_commission/July2026/channel_dicts/26August2025_Nominal_Pedestal_Periodic_Reset_Enabled_Mod0123_2025_08_26_FULL_channel_dict.json"

'''
DATADIR2="/global/cfs/cdirs/dune/users/edgarmao/trial_pedestal2/"
DATASET2_NAME="trial_pedestal2"
DATE2="2026_07_07"
DICT2_NAME="/global/cfs/cdirs/dune/users/edgarmao/2x2_crs_analysis/2x2_charge_commission/July2026/channel_dicts/trial_pedestal2_2026_07_07_FULL_channel_dict.json"
'''
DATADIR2="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/WholeDetector_20250822/"
DATASET2_NAME="WholeDetector"
DATE2="2025_08_22"
DICT2_NAME="/global/cfs/cdirs/dune/users/edgarmao/2x2_crs_analysis/2x2_charge_commission/July2026/channel_dicts/22August2025_Nominal_Pedestal_Mod0123_2025_08_22_FULL_channel_dict.json"

#DATADIR3="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/Nominal_Pedestal_256us/"
#DATASET3_NAME="Periodic_Reset_Frequency_256us_Warm"
#DATE3="2025_08_26"
#DICT3_NAME="channel_dicts/26August2025_Pedestal_Mod0123_256us_Reset_2025_08_26_FULL_channel_dict.json"

#DATADIR4="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/Nominal_Pedestal_2560us/"
#DATASET4_NAME="Periodic_Reset_Frequency_2560us_Warm"
#DATE4="2025_08_26"
#DICT4_NAME="channel_dicts/26August2025_Pedestal_Mod0123_2560us_Reset_2025_08_26_FULL_channel_dict.json"

#DATADIR5="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/Nominal_Pedestal_25600us/"
#DATASET5_NAME="Periodic_Reset_Frequency_25600us_Warm"
#DATE5="2025_08_27"
#DICT5_NAME="channel_dicts/27August2025_Pedestal_Mod0123_25600us_Reset_2025_08_27_FULL_channel_dict.json"

#DATADIR6="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/differential_LRS_1/"
#DATASET6_NAME="LRS_On_Not_Taking_Data_Warm"
#DATE6="2025_08_27"
#DICT6_NAME="channel_dicts/27August2025_Pedestal_Mod0123_LRS_On_Not_Taking_Data_2025_08_27_FULL_channel_dict.json"

#DATADIR7="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/differential_LRS_2/"
#DATASET7_NAME="LRS_On_AND_Taking_Data_Warm"
#DATE7="2025_08_28"
#DICT7_NAME="channel_dicts/28August2025_Pedestal_Mod0123_LRS_On_AND_Taking_Data_2025_08_28_FULL_channel_dict.json"

#DATADIR8="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/differential_levelMeter/"
#DATASET8_NAME="Level_Probe_On_Warm"
#DATE8="2025_08_28"
#DICT8_NAME="channel_dicts/28August2025_Pedestal_Mod0123_Level_Meter_On_2025_08_28_FULL_channel_dict.json"

#DATADIR9="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/differential_cryoPumpEnabled/"
#DATASET9_NAME="Cryo_Pump_Enabled_But_Off_Warm"
#DATE9="2025_08_29"
#DICT9_NAME="channel_dicts/29August2025_Pedestal_Mod0123_Cryo_Pump_Enabled_NOT_On_2025_08_29_FULL_channel_dict.json"

#DATADIR10="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/differential_vacuumPump/"
#DATASET10_NAME="Cryocooler_Jacket_Vacuum_Pump_On_Warm"
#DATE10="2025_08_29"
#DICT10_NAME="channel_dicts/29August2025_Pedestal_Mod0123_Vacuum_Pump_On_2025_08_29_FULL_channel_dict.json"

#DATADIR11="/global/cfs/cdirs/dune/www/data/2x2/CRS.run2/WarmComissioning/differential_PMalloff/"
#DATASET11_NAME="Purity_Monitor_All_Off_Warm"
#DATE11="2025_09_02"
#DICT11_NAME="channel_dicts/02September2025_Pedestal_Mod0123_PM_AllOff_2025_09_02_FULL_channel_dict.json"

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

python August2025_Charge_Commission_make_pedestal_plots.py -cd ${DICT1_NAME} ${DICT2_NAME} \
                                                           -n ${DATASET1_NAME} ${DATASET2_NAME} \
                                                           -d ${DATE1} ${DATE2} \
                                                           -idx ${NOM_DSET_IDX} \
                                                           -l ${YAML_MOD0} ${YAML_MOD1} ${YAML_MOD2} ${YAML_MOD3} \
                                                           -mo 0 1 2 3 \
                                                           -mm ${MAX_MEAN} \
                                                           -ms ${MAX_STD}


# SINGLE DATASET
#python August2025_Charge_Commission_make_pedestal_plots.py -cd ${DICT1_NAME} -n ${DATASET1_NAME} -d ${DATE1} -idx ${NOM_DSET_IDX} -l ${YAML_MOD0} ${YAML_MOD1} ${YAML_MOD2} ${YAML_MOD3} -mo 0 1 2 3 -mm ${MAX_MEAN} -ms ${MAX_STD}