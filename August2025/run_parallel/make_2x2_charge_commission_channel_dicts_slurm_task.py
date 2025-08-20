#!/usr/bin/env python3
import os
import sys
import glob

SLURM_NNODES = int(os.environ['SLURM_NNODES'])
SLURM_NTASKS_PER_NODE = int(os.environ['SLURM_NTASKS_PER_NODE'])
SLURM_NODEID = int(os.environ['SLURM_NODEID'])
SLURM_LOCALID = int(os.environ['SLURM_LOCALID']) # the local task ID on the node
GLOBAL_TASK_ID = SLURM_NODEID * SLURM_NTASKS_PER_NODE + SLURM_LOCALID


def get_files_in_dataset(datadir, date):
    """
    Get list of files in the dataset for a given date.
    """
    h5_files = glob.glob(f'{datadir}/*packet-N*{date}*5')
    print("Number of H5 Files:", len(h5_files), "in dataset from directory:", datadir)
    return h5_files

def main():

    dataset = sys.argv[1]
    dataset_name = sys.argv[2]
    date = sys.argv[3]

    dset_files = get_files_in_dataset(dataset, date)
    nfiles = len(dset_files)

    files_per_task = nfiles // SLURM_NTASKS_PER_NODE
    files_per_task += 1

    start_idx = GLOBAL_TASK_ID * files_per_task
    print(f"Start idx: {start_idx}")
    end_idx = start_idx + files_per_task
    print(f"End idx: {end_idx}")

    for idx in range(start_idx, end_idx):
        try:
            h5_file = dset_files[idx]
            os.system('module load python')
            os.system(f'./make_running_stats_dict.py -f {h5_file} -i {idx} -n {dataset_name} -d {date}')
        except:
            print("All files accounted for already :)")

if __name__ == '__main__':
    main()
