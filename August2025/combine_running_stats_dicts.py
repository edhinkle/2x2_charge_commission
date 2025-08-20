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


def combine_channel_running_stats_dicts(dict_dir, dataset_name, date):

    channel_full_running_stats_dict = defaultdict(lambda: {'sum': 0.0, 'sum_of_squares': 0.0, 'count': 0, 'count_invalid_parity':0})
    channel_final_stats_dict = {}

    for i, dict_file in enumerate(glob.glob(dict_dir+'/'+dataset_name+"_"+date+'_running_channel_dict*.json')):
        with open(dict_file) as running_dict_file:
            print(f"Opening dictionary file {dict_file}")
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

def main(dict_dir="run_parallel/channel_dicts", dataset_name=None, date=None):

    
    dataset_FULL_stats_dict = combine_channel_running_stats_dicts(dict_dir, dataset_name, date)

    # Save the combined dictionary to a new JSON file
    with open('channel_dicts/'+dataset_name+"_"+date+'_FULL_channel_dict.json', 'w') as outfile:
        json.dump(dataset_FULL_stats_dict, outfile, indent=4)

    #os.system(f"rm -rf {dict_dir}/{dataset_name}_{date}_running_channel_dict_*.json")

    print(f"Channel statistics dictionary for dataset '{dataset_name}' on date '{date}' saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir','--dict_dir', default="run_parallel/channel_dicts", type=str,help='''Directory with running stats dicts.''')
    parser.add_argument('-n','--dataset_name', default=None, type=str,help='''name of dataset/sample (e.g. Nominal, LRS_On)''')
    parser.add_argument('-d','--date', default=None, type=str,help='''date of data files in format YYYY_MM_DD for dataset''')
    args = parser.parse_args()
    main(**vars(args))