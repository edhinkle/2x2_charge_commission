#!/usr/bin/env python3
# Various pieces adapted from: https://github.com/larpix/crs_daq/blob/2x2-cold/analysis/plot_metric_anode.py
# In particular, plot_zy method heavily borrows from code at the above link. 

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
import glob
import matplotlib as mpl
import json
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
import yaml
import tqdm
import os

from matplotlib.axes import Axes

packet_types = {
    0: "Data",
    1: "Test",
    2: "Write",
    3: "Read",
    4: "Timestamp",
    5: "Message",
    6: "Sync",
    7: "Trigger"
}

# Helper functions relating io_group/io_channel/tile/chip/channel
def io_channel_to_tile(io_channel):
    return int(np.floor((io_channel-1-((io_channel-1)%4))/4+1))

def unique_channel_id(d):
    return ((d['io_group'].astype(int)*1000+d['io_channel'].astype(int))*1000
            + d['chip_id'].astype(int))*100 + d['channel_id'].astype(int)

def unique_to_channel_id(unique):
    return unique % 100

def unique_to_chip_id(unique):
    return (unique // 100) % 1000

def unique_to_io_channel(unique):
    return (unique//(100*1000)) % 1000

def unique_to_tiles(unique):
    return ((unique_to_io_channel(unique)-1) // 4) + 1

def unique_to_io_group(unique):
    return (unique // (100*1000*1000)) % 1000

# Geometry defaults
def _default_pxy():
    return (0., 0.)

def _rotate_pixel(pixel_pos, tile_orientation):
    return pixel_pos[0]*tile_orientation[2], pixel_pos[1]*tile_orientation[1]


# Rasterize plots 
_old_axes_init = Axes.__init__
def _new_axes_init(self, *a, **kw):
    _old_axes_init(self, *a, **kw)
    # https://matplotlib.org/stable/gallery/misc/zorder_demo.html
    # 3 => leave text and legends vectorized
    self.set_rasterization_zorder(3)
def rasterize_plots():
    Axes.__init__ = _new_axes_init
def vectorize_plots():
    Axes.__init__ = _old_axes_init
rasterize_plots()


def get_files_in_dataset(datadir, date):
    """
    Get list of files in the dataset for a given date.
    """

    #h5_files = []
    #for dir in datadirs:
    #    h5_files_new = glob.glob(f'{dir}/*packet*{date}*5')
    #    h5_files += h5_files_new
    # For now, only looking at datasets all in one directory
    h5_files = glob.glob(f'{datadir}/*packet-N*{date}*5')

    print("Number of H5 Files:", len(h5_files), "in dataset from directory:", datadir)
    return h5_files


# Dictionary saving helper methods
def tuple_key_to_string(d):
    out={}
    for key in d.keys():
        string_key=""
        max_length=len(key)
        for i in range(max_length):
            if i<len(key)-1: string_key+=str(key[i])+"-"
            else: string_key+=str(key[i])
        out[string_key]=d[key]
    return out

def int_key_to_string(d):
    out={}
    for key in d.keys():
        string_key=str(key)
        out[string_key]=d[key]
    return out                     
        
def save_dict_to_json(d, name, if_tuple):
    with open(name+".json", "w") as outfile:
        if if_tuple==True:
            updated_d = tuple_key_to_string(d)
            json.dump(updated_d, outfile, indent=4)
        else:
            updated_d = int_key_to_string(d)
            json.dump(updated_d, outfile, indent=4)

# Function to update running statistics (mean and variance)
def update_channel_dict_valid_packet_stats(stats, values):
    values = np.array(values, dtype=np.float64)  # Ensure values is a numpy array
    stats['count'] += len(values)
    if len(values) == 0:
        return stats
    # Update sum and sum of squares
    stats['sum'] += np.sum(values)
    stats['sum_of_squares'] += np.sum(np.square(values))

    return stats

# Function to update running count of invalid packets
def update_channel_dict_invalid_packet_count(stats, values):
    values = np.array(values, dtype=np.float64)  # Ensure values is a numpy array
    stats['count_invalid_parity'] += len(values)
    return stats

def make_channel_running_stats_dict_one_file(h5_file, dataset_name, date, max_entries=-1):

    channel_running_stats_dict = defaultdict(lambda: {'sum': 0.0, 'sum_of_squares': 0.0, 'count': 0, 'count_invalid_parity':0})

    print("File:", h5_file, "in Set", dataset_name)
    f = h5py.File(h5_file,'r')
    packets = f['packets']
    data_packet_mask = packets[:]['packet_type'] == 0  # Only consider data packets
    valid_parity_mask = packets[:]['valid_parity'] == 1  # Only consider packets with valid parity
    # Combine masks to filter packets
    valid_data_mask = np.logical_and(data_packet_mask, valid_parity_mask)
    adc_datawords_valid = f['packets']['dataword'][valid_data_mask][:max_entries]
    unique_ids_valid = unique_channel_id(f['packets'][valid_data_mask][:max_entries])
    unique_id_set_valid = np.unique(unique_ids_valid)

    # Update channel statistics
    for uid in tqdm.tqdm(unique_id_set_valid[:1000], desc="Looping over active channels with VALID packets ..."):
        stats = channel_running_stats_dict[uid]
        channel_mask = unique_ids_valid == uid
        channel_datawords = adc_datawords_valid[channel_mask]
        stats = update_channel_dict_valid_packet_stats(stats, channel_datawords)
        channel_running_stats_dict[uid] = stats

    # Also look at invalid parity packet count
    invalid_parity_mask = packets[:]['valid_parity'] == 0  # Packets with invalid parity
    invalid_data_mask = np.logical_and(data_packet_mask, invalid_parity_mask)
    unique_ids_invalid = unique_channel_id(f['packets'][invalid_data_mask][:max_entries])
    unique_id_set_invalid = np.unique(unique_ids_invalid)

    # Update invalid channel count
    for uid in tqdm.tqdm(unique_id_set_invalid, desc="Looping over active channels with INVALID packets ..."):
        stats = channel_running_stats_dict[uid]
        channel_mask = unique_ids_invalid == uid
        channel_invalid_packets = unique_ids_invalid[channel_mask]
        stats = update_channel_dict_invalid_packet_count(stats, channel_invalid_packets)
        channel_running_stats_dict[uid] = stats

    return channel_running_stats_dict

def combine_channel_running_stats_dicts(dict_dir, dataset_name, date):

    channel_full_running_stats_dict = defaultdict(lambda: {'sum': 0.0, 'sum_of_squares': 0.0, 'count': 0, 'count_invalid_parity':0})
    channel_final_stats_dict = {}

    for i, dict_file in enumerate(glob.glob(dict_dir+'/'+dataset_name+"_"+date+'_running_channel_dict*.json')):
        with open(dict_file) as running_dict_file:
            running_dict = json.load(running_dict_file)
            for uid, partial_stats in running_dict.items():
                full_stats = channel_full_running_stats_dict[uid]
                full_stats['count'] += partial_stats['count']
                full_stats['sum'] += partial_stats['sum']
                full_stats['sum_of_squares'] += partial_stats['sum_of_squares']
                full_stats['count_invalid_parity'] += partial_stats['count_invalid_parity']

    # Finalize channel statistics across all files
    for uid, stats in channel_full_running_stats_dict.items():
        if stats['count'] > 0:
            mean = stats['sum'] / stats['count']
            variance = (stats['sum_of_squares'] / stats['count']) - (mean ** 2)
            std = np.sqrt(variance)
            channel_final_stats_dict[uid] = {
                'mean': np.round(mean,2),
                'std': np.round(std,2), 
                'count_valid_parity': stats['count'],
                'count_invalid_parity': stats['count_invalid_parity']
            }
        else:
            channel_final_stats_dict[uid] = {
                'mean': 0.0,
                'std': 0.0, 
                'count_valid_parity': 0.0,
                'count_invalid_parity': 0.0
            }

    return channel_final_stats_dict

def main(datadirs=None, dataset_names=None, dates=None):

    num_datasets = len(dataset_names)
    
    for dset in range(num_datasets):

        # Get dataset info
        datadir = datadirs[dset]
        dataset_name = dataset_names[dset]
        date = dates[dset]

        # Get number of files in dataset
        print(f"Processing dataset '{dataset_name}' on date '{date}'...")
        h5_files_set = get_files_in_dataset(datadir, date)
        files_to_process = len(h5_files_set)

        # Set up parallelization
        files_per_task = files_to_process // SLURM_NTASKS_PER_NODE 
        if files_per_task == 0:
            files_per_task = 1
        start_idx = GLOBAL_TASK_ID * files_per_task
        end_idx = start_idx + files_per_task
        print("Start index:", start_idx)
        print("End index:", end_idx)

        # Make dictionary for each file
        for idx in range(start_idx, end_idx):
            print("File number:", idx, "in Set", dataset_name, "out of", files_to_process)

            dataset_running_stats_dict = make_channel_running_stats_dict_one_file(h5_files_set[idx], dataset_name, date, max_entries=-1)
            save_dict_to_json(dataset_running_stats_dict, "channel_dicts/"+dataset_name+"_"+date+"_running_channel_dict_"+str(idx), False)

        # Combine dictionaries
        dictionary_dir = "channel_dicts"
        dataset_FULL_stats_dict = combine_channel_running_stats_dicts(dictionary_dir, dataset_name, date)

        # Save the combined dictionary to a new JSON file
        with open(dictionary_dir+'/'+dataset_name+"_"+date+'_FULL_channel_dict.json', 'w') as outfile:
            json.dump(dataset_FULL_stats_dict, outfile, indent=4)

        os.system(f"rm -rf channel_dicts/{dataset_name}_{date}_running_channel_dict_*.json")

        print(f"Channel statistics dictionary for dataset '{dataset_name}' on date '{date}' saved.")

    print("Channel statistics dictionaries saved for all datasets.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir','--datadirs', default=None,nargs='+', type=str,help='''list of strings of data directories where data are stored for each sample.''')
    parser.add_argument('-n','--dataset_names', default=None, nargs='+', type=str,help='''name of each dataset/sample (e.g. Nominal, LRS_On)''')
    parser.add_argument('-d','--dates', default=None, nargs='+', type=str,help='''date of data files in format YYYY_MM_DD for each dataset''')
    args = parser.parse_args()
    main(**vars(args))