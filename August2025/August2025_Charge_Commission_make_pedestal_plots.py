#!/usr/bin/env python
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


def plot_xy(d, dataset_name, date, metric, list_of_geometry_yamls, module_yaml_order, norm_min, norm_max, default_pixel_pitch):

    # Set up colormap and pixel pitch
    cmap = cm.hot_r
    pixel_pitch = default_pixel_pitch

    # Define non-routed and routed channels for v2a
    nonrouted_v2a_channels = [6, 7, 8, 9, 22, 23, 24, 25, 38, 39, 40, 54, 55, 56, 57]
    routed_v2a_channels = [i for i in range(64) if i not in nonrouted_v2a_channels]

    # Set up figure and axes
    fig, ax = plt.subplots(2, 4, figsize=(40*3, 30*3))

    # Get unique channels from the dictionary
    unique_channels = np.array(list(int(key) for key in d.keys()))

    for idx in range(len(list_of_geometry_yamls)):

        geometry_yaml = list_of_geometry_yamls[idx]
        print("Geomtry yaml:", geometry_yaml)
        module = int(module_yaml_order[idx])
        print("Module:", module)

        with open(geometry_yaml) as fi:
            geo = yaml.full_load(fi)

        if 'multitile_layout_version' in geo.keys():
        # Adapted from: https://github.com/larpix/larpix-v2-testing-scripts/blob/master/event-display/evd_lib.py

            pixel_pitch = geo['pixel_pitch']

            chip_channel_to_position = geo['chip_channel_to_position']
            tile_orientations = geo['tile_orientations']
            tile_positions = geo['tile_positions']
            tpc_centers = geo['tpc_centers']
            tile_indeces = geo['tile_indeces']
            xs = np.array(list(chip_channel_to_position.values()))[
                :, 0] * pixel_pitch
            ys = np.array(list(chip_channel_to_position.values()))[
                :, 1] * pixel_pitch
            x_size = max(xs)-min(xs)+pixel_pitch
            y_size = max(ys)-min(ys)+pixel_pitch

            tile_geometry = defaultdict(int)
            io_group_io_channel_to_tile = {}
            geometry = defaultdict(_default_pxy)

            for tile in geo['tile_chip_to_io']:
                tile_orientation = tile_orientations[tile]
                tile_geometry[tile] = tile_positions[tile], tile_orientations[tile]
                for chip in geo['tile_chip_to_io'][tile]:
                    io_group_io_channel = geo['tile_chip_to_io'][tile][chip]
                    io_group = io_group_io_channel//1000
                    io_channel = io_group_io_channel % 1000
                    io_group_io_channel_to_tile[(
                        io_group, io_channel)] = tile

                for chip_channel in geo['chip_channel_to_position']:
                    chip = chip_channel // 1000
                    channel = chip_channel % 1000
                    try:
                        io_group_io_channel = geo['tile_chip_to_io'][tile][chip]
                    except KeyError:
                        print("Chip %i on tile %i in module %i not present in network" %
                              (chip, tile, module))
                        continue

                    io_group = io_group_io_channel // 1000
                    io_channel = io_group_io_channel % 1000
                    x = chip_channel_to_position[chip_channel][0] * \
                        pixel_pitch + pixel_pitch / 2 - x_size / 2
                    y = chip_channel_to_position[chip_channel][1] * \
                        pixel_pitch + pixel_pitch / 2 - y_size / 2

                    x, y = _rotate_pixel((x, y), tile_orientation)
                    x += tile_positions[tile][2] + \
                        tpc_centers[tile_indeces[tile][0]][0]
                    y += tile_positions[tile][1] + \
                        tpc_centers[tile_indeces[tile][0]][1]

                    geometry[(io_group, io_group_io_channel_to_tile[(
                        io_group, io_channel)], chip, channel)] = x, y

            xmin = min(np.array(list(geometry.values()))[:, 0])-pixel_pitch/2
            xmax = max(np.array(list(geometry.values()))[:, 0])+pixel_pitch/2
            ymin = min(np.array(list(geometry.values()))[:, 1])-pixel_pitch/2
            ymax = max(np.array(list(geometry.values()))[:, 1])+pixel_pitch/2

            if module != 2:
                tile_vertical_lines = np.linspace(xmin, xmax, 3)
                tile_horizontal_lines = np.linspace(ymin, ymax, 5)
                chip_vertical_lines = np.linspace(xmin, xmax, 21)
                chip_horizontal_lines = np.linspace(ymin, ymax, 41)

            elif module == 2:
                
                padding = np.min(np.abs(np.array(list(geometry.values()))[:, 0]))

                tile_vertical_lines = [xmin, -padding+pixel_pitch/2, padding-pixel_pitch/2, xmax]
                chip_vertical_lines = np.concatenate([np.linspace(xmin, -padding+pixel_pitch/2, 11),
                                                           np.linspace(padding-pixel_pitch/2, xmax, 11)])

                tile_horizontal_lines_mod2_temp = [-padding+pixel_pitch/2, -padding+pixel_pitch/2 - 8*10 *
                                                   pixel_pitch, -2*padding - 8*10*pixel_pitch,
                                                   -2*padding - 2*8*10*pixel_pitch]
                tile_horizontal_lines_mod2_temp2 = [padding-pixel_pitch/2, padding-pixel_pitch/2 + 8*10 * 
                                                    pixel_pitch, 2*padding + 8*10*pixel_pitch - pixel_pitch/4,
                                                    2*padding + 2*8*10*pixel_pitch - pixel_pitch/4]
                tile_horizontal_lines = np.concatenate([np.array(tile_horizontal_lines_mod2_temp), 
                                                             np.array(tile_horizontal_lines_mod2_temp2)])

                chip_horizontal_lines_mod2_temp = np.concatenate([np.linspace(-padding+pixel_pitch/2, -padding+pixel_pitch/2 - 8*10 * pixel_pitch, 11),
                                                                  np.linspace( -2*padding - 8*10*pixel_pitch, -2*padding - 2*8*10*pixel_pitch, 11),])
                chip_horizontal_lines_mod2_temp2 = np.concatenate([np.linspace(padding-pixel_pitch/2, padding-pixel_pitch/2 + 8*10 * pixel_pitch, 11),
                                                                   np.linspace(2*padding + 8*10*pixel_pitch - pixel_pitch/4, 2*padding + 2*8*10*pixel_pitch - pixel_pitch/4, 11),])

                chip_horizontal_lines = np.concatenate([np.array(chip_horizontal_lines_mod2_temp), np.array(chip_horizontal_lines_mod2_temp2)])
            else:
                raise ValueError("Module number must be 0, 1, 2, or 3.")

            for io_group in range(module*2+1, module*2+3):

                mask = unique_to_io_group(unique_channels) == io_group

                print('Getting {} for io_group {}'.format(metric, io_group))
                d_keys = unique_channels[mask]
                print('\tNumber of channels: ', len(d_keys))

                # Set up axes for the current io_group
                ax[(io_group-1) % 2, (io_group-1)//2].set_xlabel('Z Position [mm]', size=45)
                ax[(io_group-1) % 2, (io_group-1)//2].set_ylabel('Y Position [mm]', size=45)
                ax[(io_group-1) % 2, (io_group-1)//2].tick_params(axis='both', labelsize=45)

                ax[(io_group-1) % 2, (io_group-1) //2].set_xlim(xmin*1.05, xmax*1.05)
                ax[(io_group-1) % 2, (io_group-1) //2].set_ylim(ymin*1.05, ymax*1.05)

                # Draw tile and chip grid lines
                for vl in tile_vertical_lines:
                    ax[(io_group-1) % 2, (io_group-1)//2].vlines(x=vl, ymin=ymin, ymax=ymax,
                                                                 colors=['k'], linestyle='dashed')
                for hl in tile_horizontal_lines:
                    ax[(io_group-1) % 2, (io_group-1)//2].hlines(y=hl, xmin=xmin, xmax=xmax,
                                                                 colors=['k'], linestyle='dashed')
                for vl in chip_vertical_lines:
                    ax[(io_group-1) % 2, (io_group-1)//2].vlines(x=vl, ymin=ymin, ymax=ymax,
                                                                 colors=['k'], linestyle='dotted')
                for hl in chip_horizontal_lines:
                    ax[(io_group-1) % 2, (io_group-1)//2].hlines(y=hl, xmin=xmin, xmax=xmax,
                                                                 colors=['k'], linestyle='dotted')

                ax[(io_group-1) % 2, (io_group-1)//2].set_aspect('equal')

                plt.text(0.95, 1.01, 'LArPix', ha='center',
                         va='center', size=40, transform=ax[(io_group-1) % 2, (io_group-1)//2].transAxes)

                for key in d_keys:
                    channel_id = unique_to_channel_id(key)
                    chip_id = unique_to_chip_id(key)
                    tile = unique_to_tiles(key) + 8 * (1 - (io_group % 2))

                    if chip_id not in range(11, 111):
                        continue
                    if io_group != 5 and io_group != 6 and channel_id in nonrouted_v2a_channels:
                        # Modules 1, 3, 4 with v2a
                        continue
                    if channel_id not in range(64):
                        continue

                    x, y = geometry[(2 - (io_group % 2), tile,
                                         chip_id, channel_id)]
                    pitch = pixel_pitch

                    weight = (d[str(key)][metric]-norm_min)/(norm_max-norm_min)

                    if weight > 1.0:
                        weight = 1.0
                    r = Rectangle((x-(pitch/2.), y-(pitch/2.)),
                                  pitch, pitch, color=cmap(weight))
                    ax[(io_group-1) % 2, (io_group-1)//2].add_patch(r)

                colorbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(
                    vmin=norm_min, vmax=norm_max), cmap=cmap), ax=ax[(io_group-1) % 2, (io_group-1)//2])
                ax[(io_group-1) % 2, (io_group-1) //
                    2].set_title('io_group = ' + str(io_group), size=50)
                colorbar.ax.tick_params(labelsize=45)
                if metric == 'mean':
                    colorbar.set_label('Mean Dataword by Channel [ADC]', size=45)
                if metric == 'std':
                    colorbar.set_label('Standard Deviation of Dataword by Channel [ADC]', size=45)
                if metric == 'rate':
                    colorbar.set_label('Number of Hits per Channel', size=45)
    plt.suptitle('2x2 Charge Commissioning - dataword ' + metric + ' - ' + dataset_name + ' - ' + date, x=0.5, y=0.91, fontweight='bold', size=75)
    print('Saving...')
    #plt.show()
    plt.savefig('plots/2x2-xy-'+metric+'_'+dataset_name+'_'+date+'.png')
    plt.close()
    print('Saved to: plots/2x2-xy-'+metric+'_'+dataset_name+'_'+date+'.png')

def make_single_dataset_plots(channel_stats_dict, dataset_name, date, list_of_geometry_yamls, module_yaml_order, \
                              max_mean, max_std):
    """
    Make plots for a single dataset.
    """

    # Set up output PDF
    dataset_output_pdf = 'plots/'+dataset_name+'_'+date+'_2x2_WCC_analysis_summary.pdf'
    #plt.rcParams["figure.figsize"] = (10,8)
    with PdfPages(dataset_output_pdf) as output_pdf:

        mean_datawords = []
        std_datawords = []
        count_datawords = []
        for uid, stats in channel_stats_dict.items():
            chip_id = unique_to_chip_id(int(uid))
            # Exclude invalid chip ids
            if chip_id not in range(11, 111):
                continue
            # Also exclude channels with > 2000 valid packets (expect 1000)
            elif stats['count_valid_parity'] > 2000:
                continue
            # Also exclude channels with no valid packets
            elif stats['count_valid_parity'] == 0:
                continue
            else:
                mean_datawords.append(stats['mean'])
                std_datawords.append(stats['std'])
                count_datawords.append(stats['count_valid_parity'])
        mean_datawords = np.array(mean_datawords)
        std_datawords = np.array(std_datawords)
        count_datawords = np.array(count_datawords)
        num_channels = len(mean_datawords)

        dataword_bins = np.linspace(0, 255, 256)
        # Plot histogram of mean datawords
        fig, ax = plt.subplots(figsize=(8,6))
        #fig.tight_layout()
        mean_dw_counts, mean_dw_bins = np.histogram(mean_datawords, bins=dataword_bins)
        ax.hist(mean_dw_bins[:-1], bins=dataword_bins, weights=mean_dw_counts/num_channels, color='blue', alpha=0.7)
        ax.set_title(f"{dataset_name} Dataset on {date} \n [Excludes Channels with >2000 packets]", size=16)
        ax.set_xlabel('Mean Dataword Value', size=14)
        ax.set_ylabel('Fraction of Channels / ADC', size=14)
        plt.xticks(size=12)
        plt.yticks(size=12)
        ax.set_xlim(0, 52)
        ins_ax = ax.inset_axes([0.63, 0.63, 0.35, 0.35])
        ins_ax.set_title('Full Range Distribution', y=1.0, pad=-14, size=10)
        spines_to_bold = ["left", "right", "top", "bottom"]
        # Iterate through the spines and set their linewidth
        for spine_name in spines_to_bold:
            ins_ax.spines[spine_name].set_linewidth(1.5)
        ins_ax.hist(mean_dw_bins[:-1], bins=dataword_bins, weights=mean_dw_counts/num_channels, color='blue', alpha=0.9)
        #ins_ax.set_xlabel('Mean Dataword Value', size=10)
        #ins_ax.set_ylabel('Fraction of Channels', size=10)
        ins_ax.set_xlim(0, 260)
        ins_ax.set_yscale("log")
        #plt.ylim(0,1.1)
        output_pdf.savefig()
        plt.close()

        # Plot histogram of std datawords
        fig, ax = plt.subplots(figsize=(8,6))
        #fig.tight_layout()
        max_std_value = np.max(std_datawords)
        max_bin_value = int(np.ceil(max_std_value / 10) * 10)
        std_dw_counts, std_dw_bins = np.histogram(std_datawords, bins=np.linspace(0, max_bin_value, max_bin_value*4+1))
        ax.hist(std_dw_bins[:-1], bins=std_dw_bins, weights=std_dw_counts/num_channels, color='red', alpha=0.7)
        ax.set_title(f"{dataset_name} Dataset on {date} \n [Excludes Channels with >2000 packets]", size=16)
        ax.set_xlabel('Standard Deviation of Dataword Value', size=14)
        ax.set_ylabel('Fraction of Channels / 0.25 ADC', size=14)
        plt.xticks(size=12)
        plt.yticks(size=12)
        ins_ax = ax.inset_axes([0.63, 0.63, 0.35, 0.35])
        ins_ax.set_title('Full Range Distribution', y=1.0, pad=-14, size=10)
        spines_to_bold = ["left", "right", "top", "bottom"]
        # Iterate through the spines and set their linewidth
        for spine_name in spines_to_bold:
            ins_ax.spines[spine_name].set_linewidth(1.5)
        ins_ax.hist(std_dw_bins[:-1], bins=std_dw_bins, weights=std_dw_counts/num_channels, color='red', alpha=0.9)
        #ins_ax.set_xlabel('Standard Deviation of Dataword Value', size=10)
        #ins_ax.set_ylabel('Fraction of Channels / 0.25 ADC', size=10)
        ins_ax.set_xlim(-1,max(std_datawords)+5)
        ins_ax.set_yscale("log")
        ax.set_xlim(0, 4)
        output_pdf.savefig()
        plt.close()

        # Plot histogram of hits per channel
        fig, ax = plt.subplots(figsize=(10,6))
        #fig.tight_layout()
        max_count_value = np.max(count_datawords)
        max_bin_value = int(np.ceil(max_count_value / 10) * 10)
        num_bins = int(max_bin_value/100 + 1)
        count_dw_counts, count_dw_bins = np.histogram(count_datawords, bins=np.linspace(0, max_bin_value, num_bins))
        ax.hist(count_dw_bins[:-1], bins=count_dw_bins, weights=count_dw_counts/num_channels, color='green', alpha=0.7)
        ax.set_title(f"{dataset_name} Dataset on {date} \n [Excludes Channels with >2000 packets]", size=16)
        ax.set_xlabel('Valid Data Packets per Channel', size=14)
        ax.set_ylabel('Fraction of Channels / 100 Valid Data Packets', size=14)
        plt.xticks(size=12)
        plt.yticks(size=12)
        ax.set_xlim(0)
        ax.set_yscale('log')
        output_pdf.savefig()
        plt.close()

    plot_xy(d=channel_stats_dict, dataset_name=dataset_name, date=date, metric='mean', \
            list_of_geometry_yamls=list_of_geometry_yamls, module_yaml_order=module_yaml_order, \
            norm_min=0, norm_max=max_mean, default_pixel_pitch=4.4)
    plot_xy(d=channel_stats_dict, dataset_name=dataset_name, date=date, metric='std', \
            list_of_geometry_yamls=list_of_geometry_yamls, module_yaml_order=module_yaml_order, \
            norm_min=0, norm_max=max_std, default_pixel_pitch=4.4)
    return 0

# Helper method to create differential dictionarys
def make_differential_dict(nominal_dict=None, secondary_dict=None):
    ''' Method to store difference in channel mean/standard deviation/valid parity packet count/invalid
        parity packet count between a 'nominal' dataset and another dataset '''
    diff_dict = {}

    # Open nominal dictionary
    nominal_dict_file = nominal_dict
    nominal_dict_file_open = open(nominal_dict_file)
    nominal_dict = json.load(nominal_dict_file_open)

    # Open secondary dictionary
    secondary_dict_file = secondary_dict
    secondary_dict_file_open = open(secondary_dict_file)
    secondary_dict = json.load(secondary_dict_file_open)

    # Check keys in nominal dict
    for key in nominal_dict.keys():
        if key in secondary_dict:
            diff_dict[key] = {
                'mean': secondary_dict[key]['mean'] - nominal_dict[key]['mean'],
                'std': secondary_dict[key]['std'] - nominal_dict[key]['std'],
                'count_valid_parity': secondary_dict[key]['count_valid_parity'] - nominal_dict[key]['count_valid_parity'],
                'count_invalid_parity': secondary_dict[key]['count_invalid_parity'] - nominal_dict[key]['count_invalid_parity']
            }
        elif key not in secondary_dict:
            diff_dict[key] = {
                'mean': -nominal_dict[key]['mean'],
                'std': -nominal_dict[key]['std'],
                'count_valid_parity': -nominal_dict[key]['count_valid_parity'],
                'count_invalid_parity': -nominal_dict[key]['count_invalid_parity']
            }

    # Check keys in secondary dict
    for key in secondary_dict.keys():
        if key not in nominal_dict:
            diff_dict[key] = {
                'mean': secondary_dict[key]['mean'],
                'std': secondary_dict[key]['std'],
                'count_valid_parity': secondary_dict[key]['count_valid_parity'],
                'count_invalid_parity': secondary_dict[key]['count_invalid_parity']
            }

    return diff_dict

# Helper method to create plots for differential datasets
def make_differential_plots(diff_dict=None, nominal_dataset_name=None, nominal_dataset_date=None, 
                            secondary_dataset_name=None, secondary_dataset_date=None, list_of_geometry_yamls=None, 
                            module_yaml_order=None, max_mean=None, max_std=None):
    
    min_mean_diff = -(max_mean / 2.)
    print("Min mean diff:", min_mean_diff)
    max_mean_diff = max_mean / 2.
    min_std_diff = -(max_std / 2.)
    print("Max mean std: ", min_std_diff)
    max_std_diff = max_std / 2.

    # Set up output PDF
    dataset_output_pdf = 'plots/'+secondary_dataset_name+'_'+secondary_dataset_date+'_MINUS_'+nominal_dataset_name+'_'+nominal_dataset_date+'_2x2_WCC_differential_analysis_summary.pdf'
    #plt.rcParams["figure.figsize"] = (10,8)
    with PdfPages(dataset_output_pdf) as output_pdf:

        mean_dataword_diff = []
        std_dataword_diff = []
        count_dataword_diff = []
        for uid, stats in diff_dict.items():
            chip_id = unique_to_chip_id(int(uid))
            # Exclude invalid chip ids
            if chip_id not in range(11, 111):
                continue
            else:
                mean_dataword_diff.append(stats['mean'])
                std_dataword_diff.append(stats['std'])
                count_dataword_diff.append(stats['count_valid_parity'])
        mean_dataword_diff = np.array(mean_dataword_diff)
        std_dataword_diff = np.array(std_dataword_diff)
        count_dataword_diff = np.array(count_dataword_diff)
        num_channels = len(mean_dataword_diff)

        # Plot histogram of mean dataword difference
        fig, ax = plt.subplots(figsize=(8,6))
        #fig.tight_layout()
        dataword_mean_bins = np.linspace(-256, 256, 513)
        mean_dw_counts, mean_dw_bins = np.histogram(mean_dataword_diff, bins=dataword_mean_bins)
        ax.hist(mean_dw_bins[:-1], bins=dataword_mean_bins, weights=mean_dw_counts/num_channels, color='blue', alpha=0.7)
        ax.set_title(f"{secondary_dataset_name} Dataset on {secondary_dataset_date} \n MINUS {nominal_dataset_name} Dataset on {nominal_dataset_date}", size=15)
        ax.set_xlabel('Difference in Mean Dataword Value', size=14)
        ax.set_ylabel('Fraction of Channels / ADC', size=14)
        plt.xticks(size=12)
        plt.yticks(size=12)
        ax.set_xlim(min_mean_diff, max_mean_diff)
        ins_ax = ax.inset_axes([0.63, 0.63, 0.35, 0.35])
        ins_ax.set_title('Full Range Distribution', y=1.0, pad=-14, size=10)
        spines_to_bold = ["left", "right", "top", "bottom"]
        # Iterate through the spines and set their linewidth
        for spine_name in spines_to_bold:
            ins_ax.spines[spine_name].set_linewidth(1.5)
        ins_ax.hist(mean_dw_bins[:-1], bins=dataword_mean_bins, weights=mean_dw_counts/num_channels, color='blue', alpha=0.9)
        #ins_ax.set_xlabel('Mean Dataword Value', size=10)
        #ins_ax.set_ylabel('Fraction of Channels', size=10)
        #ins_ax.set_xlim(0, 260)
        ins_ax.set_yscale("log")
        #plt.ylim(0,1.1)
        output_pdf.savefig()
        plt.close()

        # Plot histogram of std datawords
        fig, ax = plt.subplots(figsize=(8,6))
        #fig.tight_layout()
        max_std_value = np.max(std_dataword_diff)
        max_bin_value = int(np.ceil(max_std_value / 10) * 10)
        min_std_value = np.min(std_dataword_diff)
        min_bin_value = int(np.floor(min_std_value / 10) * 10)
        std_dw_counts, std_dw_bins = np.histogram(std_dataword_diff, bins=np.linspace(min_bin_value, max_bin_value, (max_bin_value - min_bin_value)*4+1))
        ax.hist(std_dw_bins[:-1], bins=std_dw_bins, weights=std_dw_counts/num_channels, color='red', alpha=0.7)
        ax.set_title(f"{secondary_dataset_name} Dataset on {secondary_dataset_date} \n MINUS {nominal_dataset_name} Dataset on {nominal_dataset_date}", size=15)
        ax.set_xlabel('Difference in Standard Deviation of Dataword Value', size=14)
        ax.set_ylabel('Fraction of Channels / 0.25 ADC', size=14)
        plt.xticks(size=12)
        plt.yticks(size=12)
        ins_ax = ax.inset_axes([0.63, 0.63, 0.35, 0.35])
        ins_ax.set_title('Full Range Distribution', y=1.0, pad=-14, size=10)
        spines_to_bold = ["left", "right", "top", "bottom"]
        # Iterate through the spines and set their linewidth
        for spine_name in spines_to_bold:
            ins_ax.spines[spine_name].set_linewidth(1.5)
        ins_ax.hist(std_dw_bins[:-1], bins=std_dw_bins, weights=std_dw_counts/num_channels, color='red', alpha=0.9)
        #ins_ax.set_xlabel('Standard Deviation of Dataword Value', size=10)
        #ins_ax.set_ylabel('Fraction of Channels / 0.25 ADC', size=10)
        ins_ax.set_xlim(min(std_dataword_diff-5),max(std_dataword_diff)+5)
        ins_ax.set_yscale("log")
        ax.set_xlim(min_std_diff, max_std_diff)
        output_pdf.savefig()
        plt.close()

        # Plot histogram of hits per channel
        fig, ax = plt.subplots(figsize=(10,6))
        #fig.tight_layout()
        max_count_value = np.max(count_dataword_diff)
        max_bin_value = int(np.ceil(max_count_value / 10) * 10)
        min_count_value = np.min(count_dataword_diff)
        min_bin_value = int(np.floor(min_count_value / 10) * 10)
        num_bins = int((max_bin_value-min_bin_value)/100 + 1)
        count_dw_counts, count_dw_bins = np.histogram(count_dataword_diff, bins=np.linspace(min_bin_value, max_bin_value, num_bins))
        ax.hist(count_dw_bins[:-1], bins=count_dw_bins, weights=count_dw_counts/num_channels, color='green', alpha=0.7)
        ax.set_title(f"{secondary_dataset_name} Dataset on {secondary_dataset_date} \n MINUS {nominal_dataset_name} Dataset on {nominal_dataset_date} \n [-4000, 4000]", size=15)
        ax.set_xlabel('Difference in Valid Data Packets per Channel', size=14)
        ax.set_ylabel('Fraction of Channels / 100 Valid Data Packets', size=14)
        plt.xticks(size=12)
        plt.yticks(size=12)
        ax.set_xlim(-4000, 4000)
        ax.set_yscale('log')
        output_pdf.savefig()
        plt.close()

    dataset_name = secondary_dataset_name + "__MINUS__" + nominal_dataset_name
    date = "Set1_" + secondary_dataset_date + "_Set2_" + nominal_dataset_date

    plot_xy(d=diff_dict, dataset_name=dataset_name, date=date, metric='mean', \
            list_of_geometry_yamls=list_of_geometry_yamls, module_yaml_order=module_yaml_order, \
            norm_min=min_mean_diff, norm_max=max_mean_diff, default_pixel_pitch=4.4)
    plot_xy(d=diff_dict, dataset_name=dataset_name, date=date, metric='std', \
            list_of_geometry_yamls=list_of_geometry_yamls, module_yaml_order=module_yaml_order, \
            norm_min=min_std_diff, norm_max=max_std_diff, default_pixel_pitch=4.4)


    return 0


def main(channel_dicts=None, dataset_names=None, dates=None, nominal_dataset_idx=0,\
         list_of_geometry_yamls=None, module_yaml_order=None, max_mean=None, max_std=None):

    # Check number of datasets
    num_dsets = len(channel_dicts)

    for dset in tqdm.tqdm(range(num_dsets), desc="Making single dataset plots ..."):

        # Load channel dictionary
        channel_dict_file = channel_dicts[dset]
        channel_dict_file_open = open(channel_dict_file)
        channel_dict = json.load(channel_dict_file_open)
        dataset_name = dataset_names[dset]
        date = dates[dset]

        print(f"Processing dataset '{dataset_name}' on date '{date}'...")

        make_single_dataset_plots(channel_dict, dataset_name, date, list_of_geometry_yamls, module_yaml_order, \
                              max_mean, max_std)

    # Compare datasets if multiple are given
    if num_dsets > 1:
        print("Comparing datasets...")
        nominal_dataset_name = dataset_names[nominal_dataset_idx]
        nominal_dataset_date = dates[nominal_dataset_idx]
        for dset in tqdm.tqdm(range(num_dsets), desc="Making differential dictionaries and plots ..."):
            if dset != nominal_dataset_idx:
                diff_dict = make_differential_dict(channel_dicts[nominal_dataset_idx], channel_dicts[dset])
                make_differential_plots(diff_dict=diff_dict,nominal_dataset_name=nominal_dataset_name, 
                                        nominal_dataset_date=nominal_dataset_date, 
                                        secondary_dataset_name=dataset_names[dset], 
                                        secondary_dataset_date=dates[dset], 
                                        list_of_geometry_yamls=list_of_geometry_yamls, 
                                        module_yaml_order=module_yaml_order, max_mean=max_mean, max_std=max_std)
            else: continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cd','--channel_dicts', default=None,nargs='+', type=str,help='''list of strings of channel dictionary locations for each sample.''')
    parser.add_argument('-n','--dataset_names', default=None, nargs='+', type=str,help='''name of each dataset/sample (e.g. Nominal, LRS_On)''')
    parser.add_argument('-d','--dates', default=None, nargs='+', type=str,help='''date of data files in format YYYY_MM_DD for each dataset''')
    parser.add_argument('-idx','--nominal_dataset_idx', default=0, type=int, help='''index of the nominal dataset in list of datasets in datadirs/dataset_names/dates''')
    parser.add_argument('-l','--list_of_geometry_yamls', default=None, nargs='+', type=str, help='''list of strings giving paths to geometry yamls for each module''')
    parser.add_argument('-mo', '--module_yaml_order', default=None, nargs='+', type=int, help='''list of ints giving module numbers for order of geometry yamls in list of geometry yamls''')
    parser.add_argument('-mm', '--max_mean', default=50, type=int, help='''Max ADC mean for anode view mean ADC plots''')
    parser.add_argument('-ms', '--max_std', default=5, type=int, help='''Max standard deviation of ADC values for anode view std ADC plots''')
    #parser.add_argument('-bd', '--binarydirs', default=None,nargs='+', type=str,help='''directories where binary files are stored''')
    args = parser.parse_args()
    main(**vars(args))