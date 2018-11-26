import numpy as np
import sys
import os
import shutil
import suite2p
from suite2p.run_s2p import run_s2p

# Define relevant subfolders
Ffast = '~/Desktop/Test'

if os.path.isdir(Ffast):
    shutil.rmtree(Ffast)
os.mkdir(Ffast)

Ftif  = '/Volumes/MARIANNE/BL'

# Define options for running suite2p
#-------------------------------------------------------------------------------
ops = {
       # file paths
       'look_one_level_down': False, # whether to look in all subfolders when searching for tiffs
       'fast_disk': Ffast, # used to store temporary binary file, defaults to save_path0 (set to a string NOT a list)
       'delete_bin': True, # whether to delete binary file after processing
       'h5py_key': 'data', # key in h5 where data array is stored (data should be time x pixels x pixels)

       # main settings
       'nplanes' : 10, # each tiff has these many planes in sequence
       'nchannels' : 1, # each tiff has these many channels per plane
       'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
       'diameter':2, # this is the main parameter for cell detection, 2-dimensional if Y and X are different (e.g. [6 12])
       'tau':  1., # this is the main parameter for deconvolution
       'fs': 4.,  # sampling rate (total across planes)

       # output settings
       'save_mat': False, # whether to save output as matlab files
       'combined': True, # combine multiple planes into a single result /single canvas for GUI

       # parallel settings
       'num_workers': 0, # 0 to select num_cores, -1 to disable parallelism, N to enforce value
       'num_workers_roi': -1, # 0 to select number of planes, -1 to disable parallelism, N to enforce value

       # registration settings
       'do_registration': False, # whether to register data
       'nimg_init': 50, # subsampled frames for finding reference image
       'batch_size': 200, # number of frames per batch
       'maxregshift': 0.1, # max allowed registration shift, as a fraction of frame max(width and height)
       'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
       'reg_tif': False, # whether to save registered tiffs
       'subpixel' : 10, # precision of subpixel registration (1/subpixel steps)

       # cell detection settings
       'connected': True, # whether or not to keep ROIs fully connected (set to 0 for dendrites)
       'navg_frames_svd': 5000, # max number of binned frames for the SVD
       'nsvd_for_roi': 1000, # max number of SVD components to keep for ROI detection
       'max_iterations': 20, # maximum number of iterations to do cell detection
       'ratio_neuropil': 2., # ratio between neuropil basis size and cell radius
       'ratio_neuropil_to_cell': 1, # minimum ratio between neuropil radius and cell radius
       'tile_factor': 1., # use finer (>1) or coarser (<1) tiles for neuropil estimation during cell detection
       'threshold_scaling': 1., # adjust the automatically determined threshold by this scalar multiplier
       'max_overlap': 0.75, # cells with more overlap than this get removed during triage, before refinement
       'inner_neuropil_radius': 2, # number of pixels to keep between ROI and neuropil donut
       'outer_neuropil_radius': np.inf, # maximum neuropil radius
       'min_neuropil_pixels': 50, # minimum number of pixels in the neuropil

       # deconvolution settings
       'baseline': 'maximin', # baselining mode
       'win_baseline': 60., # window for maximin
       'sig_baseline': 10., # smoothing constant for gaussian filter
       'prctile_baseline': 8.,# optional (whether to use a percentile baseline)
       'neucoeff': .7,  # neuropil coefficient
       'allow_overlap': False,
       'xrange': np.array([0, 0]),
       'yrange': np.array([0, 0]),
     }

# Provide data path
#-------------------------------------------------------------------------------
db = {
    'h5py': [],
    'hp5y_key': 'data',
    'look_one_level_down': False,
    'data_path': [Ftif],
    'fast_disk': Ffast
}

# Run experiment
#-------------------------------------------------------------------------------
opsEnd = run_s2p(ops=ops, db=db)
