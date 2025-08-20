#!/usr/bin/env python

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
import glob
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages


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

def io_channel_to_tile(io_channel):
    return int(np.floor((io_channel-1-((io_channel-1)%4))/4+1))

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

def main(datadirs=None, date=None, binarydirs=None):

    h5_files = []
    for dir in datadirs:

        h5_files_new = glob.glob(f'{dir}/packet*{date}*5')
        h5_files += h5_files_new

    print("Number of H5 Files:", len(h5_files))
    #print(h5_files)
    time_only_list = []
    for file in h5_files:
        file_name = file.split("/")[-1]
        date_only = file_name.split("_CDT")[0]
        time_only = date_only.split(date+'_')[-1]
        time_only_list.append(time_only)

    # Pair each string with its original position
    indexed_time_only = list(enumerate(time_only_list))

    # Sort the pairs by string
    sorted_pairs = sorted(indexed_time_only, key=lambda pair: pair[1])

    # Separate the positions and sorted strings
    time_positions, sorted_strings = zip(*sorted_pairs)
    no_config_files = []
    with open(date+"_file_information.txt", "a") as text_file:
        for i in range(len(h5_files)):

            #if i!=0: continue
            if i%10==0: print("File number:", i)
            config_info = ''
            file_number = time_positions[i]
            file = h5_files[file_number]
            print("\n------------------------------------", file=text_file)
            print("File:", file, file=text_file)
            for dir in binarydirs:
                binary_files = glob.glob(f'{dir}/*{date}*.h*5')
                for binary_file in binary_files:
                    if sorted_strings[i] in binary_file:
                        binary_file_name = binary_file
                    else: continue
            print("Binary File:", binary_file_name, file=text_file)
            bf = h5py.File(binary_file_name,'r')
            #print('Keys in binary file:',list(bf.keys()))
            if 'daq_configs' not in bf.keys():
                print("DAQ Configs not found in binary file for data file timestamped at "+\
                          date+' '+sorted_strings[i]+" CDT. Moving onto next data file.", file=text_file)
                no_config_files.append(sorted_strings[i])
                config_info = ' NO CONFIG INFO'
            
            f = h5py.File(file,'r')
            print('Keys in file:',list(f.keys()), file=text_file)
            plt.rcParams["figure.figsize"] = (10,8)
            for key in f.keys():
                print('Number of',key,'entries in file:', len(f[key]), file=text_file)
            output_pdf_name = 'plots/'+date+'_'+sorted_strings[i]+'_CDT_analysis_plots.pdf'
            print("IMPORTANT NOTE: Only IO Groups 1 and 2 are considered.", file=text_file)
            # put file in this directory for now
            with PdfPages(output_pdf_name) as output:

                # Check for packets with dataword==128
                weird_dw = np.where(np.array(f['packets']['dataword']) == 128)
                datawords = f['packets']['dataword'][weird_dw]
                timestamps = f['packets']['timestamp'][weird_dw]
                iogs = f['packets']['io_group'][weird_dw]
                check_iog = np.where((np.array(iogs) == 1) | (np.array(iogs) == 2))
                datawords = datawords[check_iog]
                timestamps = timestamps[check_iog]
                max_ts, min_ts = np.max(f['packets']['timestamp']), np.min(f['packets']['timestamp'])
                chip_ids = f['packets']['chip_id'][weird_dw]
                chip_ids = chip_ids[check_iog]
                print("Packet types represented in file:", np.unique(np.array(f['packets']['packet_type'])), file=text_file)
                print("Number of Packets with Dataword==128:", len(datawords), file=text_file)
                fig = plt.figure(figsize=(10,10))
                fig.tight_layout()
                plt.scatter(timestamps, chip_ids, color='green', s=1)
                plt.title("Chip ID vs Timestamp for Packets with Dataword==128 for\n IO Groups 1&2 in file timestamped at "+date+' '+sorted_strings[i]+" CDT"+config_info)
                plt.xlabel('Timestamp [CRS Ticks]')
                plt.xlim(min_ts, max_ts)
                #plt.ylim(0,120)
                plt.axhline(y=10.9, color='r', linestyle='--')
                plt.axhline(y=110.1, color='r', linestyle='--')
                plt.ylabel('Chip ID')
                output.savefig()
                plt.close()

                # Check packet parity
                fig = plt.figure(figsize=(10,10))
                fig.tight_layout()
                ts_bins = np.linspace(min_ts, max_ts, 101)
                check_packet_parity = np.where(np.array(f['packets']['valid_parity']) != 1)
                iog_packet_parity = f['packets']['io_group'][check_packet_parity]
                invalid_parity_timestamps = f['packets']['timestamp'][check_packet_parity]
                check_iog = np.where((np.array(iog_packet_parity) == 1) | (np.array(iog_packet_parity) == 2))
                invalid_parity_timestamps = invalid_parity_timestamps[check_iog]
                print("Number of Packets with Invalid Parity:", len(invalid_parity_timestamps), file=text_file)
                plt.hist(invalid_parity_timestamps, bins=ts_bins, color='blue')
                plt.title("Timestamps of Packets with Invalid Parity for\n IO Groups 1&2 in file timestamped at "+date+' '+sorted_strings[i]+" CDT"+config_info)
                plt.xlabel('Timestamp [CRS Ticks]')
                plt.xlim(min_ts, max_ts)
                plt.ylabel('Number of Packets / '+str((max_ts-min_ts)/100)+' CRS Ticks')
                plt.yscale('log')
                output.savefig()
                plt.close()

                # Check for non-data packets
                fig = plt.figure(figsize=(10,10))
                fig.tight_layout()
                check_non_data = np.where(np.array(f['packets']['packet_type']) != 0)
                non_data_timestamps = f['packets']['timestamp'][check_non_data]
                iogs_non_data = f['packets']['io_group'][check_non_data]
                packet_types_non_data = f['packets']['packet_type'][check_non_data]
                check_iog = np.where((np.array(iogs_non_data) == 1) | (np.array(iogs_non_data) == 2))
                non_data_timestamps = non_data_timestamps[check_iog]
                packet_types_non_data = packet_types_non_data[check_iog]
                print("Number of Non-Data Packets:", len(non_data_timestamps), file=text_file)
                unique_packet_types = np.unique(np.array(packet_types_non_data))
                unique_packet_type_labels = [packet_types[pt] for pt in unique_packet_types]
                print("Unique Packet Types for Non-Data Packets:", unique_packet_type_labels, file=text_file)
                for pt in np.unique(np.array(packet_types_non_data)):
                    pt_timestamps = non_data_timestamps[np.where(packet_types_non_data == pt)]
                    plt.hist(pt_timestamps, bins=ts_bins, label=packet_types[pt], alpha=0.7)
                plt.title("Non-Data Packet Type vs Timestamp for Non-Data Packets for\n IO Groups 1&2 in file timestamped at "+date+' '+sorted_strings[i]+" CDT"+config_info)
                plt.xlabel('Timestamp [CRS Ticks]')
                plt.xlim(min_ts, max_ts)
                plt.ylabel('Number of Packets / '+str((max_ts-min_ts)/100)+' CRS Ticks')
                plt.yscale('log')
                legend = plt.legend(title='Packet Type')
                plt.setp(legend.get_title(), weight='bold')
                output.savefig()
                plt.close()

                # Check packet parity by ASIC
                fig = plt.figure(figsize=(20,20))
                fig.tight_layout()
                ax = []
                packets_iogroup_inv_parity = f['packets']['io_group'][check_packet_parity]
                packets_iochannel_inv_parity = f['packets']['io_channel'][check_packet_parity]
                packets_tiles_inv_parity = np.array([io_channel_to_tile(io_channel) for io_channel in packets_iochannel_inv_parity])
                packets_chip_id_inv_parity = f['packets']['chip_id'][check_packet_parity]
                values, counts = np.unique(packets_chip_id_inv_parity, return_counts=True)
                norm = mpl.colors.LogNorm(vmin=1,vmax=max(counts))
                cmap = plt.colormaps.get_cmap('viridis')
                mcharge = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                for iog in range(2):
                    check_packets_in_iog = np.where(np.array(packets_iogroup_inv_parity) == iog+1)
                    tiles_in_iog_invalid_parity = packets_tiles_inv_parity[check_packets_in_iog]
                    chip_ids_in_iog_invalid_parity = packets_chip_id_inv_parity[check_packets_in_iog]
                    if len(np.array(tiles_in_iog_invalid_parity)) == 0:
                        print("None of the Packets for IO Group "+str(iog+1)+" have invalid parity.", file=text_file)
                        continue
                    max_tile_id, min_tile_id = np.max(tiles_in_iog_invalid_parity), np.min(tiles_in_iog_invalid_parity)
                    max_chip_id, min_chip_id = np.max(chip_ids_in_iog_invalid_parity), np.min(chip_ids_in_iog_invalid_parity)
                    tile_bins = np.linspace(min_tile_id, max_tile_id, max_tile_id-min_tile_id+1)
                    chip_id_bins = np.linspace(min_chip_id, max_chip_id, max_chip_id-min_chip_id+1)
                    ax.append(fig.add_subplot(1,2,iog+1))
                    ax[iog].hist2d(tiles_in_iog_invalid_parity, chip_ids_in_iog_invalid_parity, bins=[tile_bins, chip_id_bins], \
                                   cmap=cmap, norm=norm)
                    ax[iog].set_title("IO Group "+str(iog+1), size=20)
                    ax[iog].set_xlabel('Tile ID', size=16)
                    ax[iog].set_xlim(0,8)
                    ax[iog].tick_params(axis='x', size=16)
                    ax[iog].set_ylabel('Chip ID', size=16)
                    ax[iog].set_ylim(min_chip_id, max(max_chip_id, 115))
                    ax[iog].tick_params(axis='y', size=16)
                    #ax[iog].set_yscale('log')
                    print("Packets with Invalid Parity for IO Group "+str(iog+1)+":", len(tiles_in_iog_invalid_parity), file=text_file)
                fig.subplots_adjust(right=0.90)
                cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.75])
                cbar = fig.colorbar(mcharge, cax=cbar_ax, label='Number of Packets')
                cbar.set_label(r'Number of Packets', size=16)
                cbar_ax.tick_params(labelsize=14)
                plt.suptitle("Identifying ASICs for Packets with Invalid Parity for\n IO Groups 1&2 in file timestamped at "+date+' '+sorted_strings[i]+" CDT"+config_info, size=24)
                output.savefig()
                plt.close()

            print('------------------------------------------------\n', file=text_file)
        print("No config datasets found for the following timestamps:",no_config_files, file=text_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd','--datadirs', default=None,nargs='+', type=str,help='''list of strings of data directories where data are stored.''')
    parser.add_argument('-d','--date', default=None, type=str,help='''date of data files in format YYYY_MM_DD''')
    parser.add_argument('-bd', '--binarydirs', default=None,nargs='+', type=str,help='''directories where binary files are stored''')
    args = parser.parse_args()
    main(**vars(args))