"""
Extract electrode locations from probe mapping output of HERBS

Created on Mon Jun  5 17:10:29 2023

@author: Greg Knoll
"""

import numpy as np
import pandas as pd
import joblib
from bs4 import BeautifulSoup

# from neuropixels_preprocessing.session_params import *

#If histology of rat isn't available yet, estimate brain areas and create sham "Rx_probe1_electrode_locations.csv" file

# -------------------------------------------------------------------------- #
#                   Load unit electrode numbers
# -------------------------------------------------------------------------- #

# In Phy, electrodes are referred to (unfortunately) as channels
# Create a database of units and their "channels"

DATA_ROOT = 'local'   # ['local', 'server', 'X:', 'Y:', 'D:']
rat = '3'
date = '20231110'
probe_num = 1
actual_histology = True    # set to False if using estimated electrode-brain area assignments

root_path = get_root_path(DATA_ROOT)

if not actual_histology:
    area_labels = ["Prelimbic_area", "Secondary_motor_area"]
    area_sites = [305, 77]

metadata = load_session_metadata_from_csv(DATA_ROOT, rat, date)
trodes_datetime = metadata['trodes_datetime']
ROOT_DATAPATH = f'{root_path}{rat}/ephys/{trodes_datetime}.rec/'
PREPROCESS_DIR = ROOT_DATAPATH + f'preprocessing_output/probe{probe_num}/'
SORTER_DIR = ROOT_DATAPATH + f'sorting_output/probe{probe_num}/sorter_output/'

session_data = joblib.load(PREPROCESS_DIR + 'spike_mat_in_ms.npy')
unit_list = session_data['row_cluster_id']
cl_df = pd.read_csv(SORTER_DIR + 'cluster_info.tsv', sep="\t")
unit_channel_df = cl_df[cl_df['cluster_id'].isin(unit_list)][['cluster_id', 'ch']].reset_index(drop=True)
assert np.all(unit_channel_df['group']=='good')
assert len(unit_channel_df) == len(unit_list)
assert unit_channel_df['ch'][unit_channel_df['cluster_id']==unit_list[0]].values == \
        cl_df['ch'][cl_df['cluster_id']==unit_list[0]].values

# phy_channel_map = np.load(SESSION_DIR + 'channel_map.npy')[0]
# check that phy simply uses the active electrodes in order
# assert np.array_equal(np.arange(0, n_active_electrodes), phy_channel_map)

# Now add the actual electrode number, based on the "channel"
# If all electrodes are mapped consecutively to "channel", then the electrode
# number is just the channel number with an offset

Trodes_config = ROOT_DATAPATH + metadata['trodes_config'] + f'.trodesconf'
with open(Trodes_config, 'r') as f:
    data = f.read()
Bs_data = BeautifulSoup(data, "xml")

channelsOn_str = Bs_data.find_all('CustomOption', attrs={'name':'channelsOn'})[probe_num-1].get('data')
channelsOn_arr = np.fromstring(channelsOn_str, sep=' ')
# assert channelsOn_arr.size == n_Npix1_electrodes
active_electrodes = np.where(channelsOn_arr)[0]
# assert len(active_electrodes) == n_active_electrodes
unit_channel_df['electrode'] = active_electrodes[unit_channel_df.ch]
# -------------------------------------------------------------------------- #


# -------------------------------------------------------------------------- #
#         Find which electrodes correspond to which brain region
# -------------------------------------------------------------------------- #
if actual_histology:
    PROBE_PKL_PATH = f'{DATA_ROOT}/Neurodata/{rat}/ephys/{trodes_datetime}.rec/preprocessing_output/probe{probe_num}/'
    probe_pkl = pd.read_pickle(PROBE_PKL_PATH + f'probe{probe_num}.pkl')
    probe_data = probe_pkl['data']
    probe_df = pd.DataFrame(columns=['Region', 'Lowest Electrode', 'Highest Electrode'])


    for ROI_idx, ROI in enumerate(probe_data['label_name']):
        ROI_start = sum(probe_data['region_sites'][:ROI_idx])+1
        ROI_end = sum(probe_data['region_sites'][:ROI_idx+1])
        new_row = pd.DataFrame([[ROI, int(ROI_start), int(ROI_end)]], columns=probe_df.columns)

        probe_df = pd.concat([probe_df, new_row], ignore_index=True)

    filename = f'R{rat}_probe{probe_num}_electrode_locations.csv'
    probe_df.to_csv(PROBE_PKL_PATH + filename, index=False, header=True)
    probe_df.to_csv(PREPROCESS_DIR + filename, index=False, header=True)
else:
    probe_data = {"label_name": area_labels, "region_sites": area_sites}
# -------------------------------------------------------------------------- #


# -------------------------------------------------------------------------- #
#                   Add region to DataFrame and save
# -------------------------------------------------------------------------- #
unit_channel_df['region'] = ""
for ROI_idx, ROI in enumerate(probe_data['label_name']):
    ROI_start = sum(probe_data['region_sites'][:ROI_idx])+1
    ROI_end = sum(probe_data['region_sites'][:ROI_idx+1])
    ROI_units = (ROI_start < unit_channel_df.electrode) & (unit_channel_df.electrode <= ROI_end)
    unit_channel_df.loc[ROI_units, 'region'] = ROI

unit_channel_df.to_csv(PREPROCESS_DIR + f'{rat}_{date}_probe{probe_num}--good_unit_stats_w_region.tsv', sep='\t', index=False)
