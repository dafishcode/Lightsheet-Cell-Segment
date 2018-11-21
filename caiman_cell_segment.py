################################################################################
## Code adapted from demo_caiman_cnmf_3D as imported from github 21/11/2018
## https://github.com/flatironinstitute/CaImAn
################################################################################

import logging
import matplotlib as plt
import numpy as np
import os
import psutil
from scipy.ndimage.filters import gaussian_filter
import sys
import re

import caiman as cm
from caiman.utils.visualization import nb_view_patches3d
import caiman.source_extraction.cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.components_evaluation import evaluate_components, estimate_components_quality_auto
from caiman.cluster import setup_cluster
from caiman.paths import caiman_datadir

import bokeh.plotting as bpl

# Housekeeping
#===============================================================================
# Module flags
display_movie   = False      # play movie of tifs that are loaded
save_results    = False      # flag to save results or not

# Define folder locations
#-------------------------------------------------------------------------------
Fbase  = '/Volumes/GoogleDrive/My Drive/Research/1809 Seizure criticality/data/Zebrafish/ZFRR007-01'
Fmvct  = Fbase + os.sep + 'Realigned'
Fshort = Fmvct + os.sep + 'Test'

# Load tifs
#-------------------------------------------------------------------------------
Fthis   = Fshort
tifs    = []
ftifs   = os.listdir(Fthis)
r       = re.compile('^.*[tif|tiff]$')
ftifs   = list(filter(r.match, ftifs))
for ff in ftifs: tifs.append(os.path.join(Fthis, ff))
print("I found " + str(len(tifs)) + " tif files")

# Segmentation parameter settings
#===============================================================================
# Set up parallel processing
#-------------------------------------------------------------------------------
n_processes = psutil.cpu_count()
print('using ' + str(n_processes) + ' processes')
print("Stopping  cluster to avoid unnencessary use of memory....")
sys.stdout.flush()
cm.stop_server()

c, dview, n_processes = cm.cluster.setup_cluster(backend = 'local', n_processes = None, single_thread = False)

# Set Parameters (taken from standard demo)
#-------------------------------------------------------------------------------
fr          = 4                     # imaging rate in frames per seconds
decay_time  = .4                    # length of a typical transient in seconds

rf      = (15,15,15)                # halfsize of patches in pixels
stride  = (10,10,10)                # amount of overlap between patches in pixels
K       = 12                        # number of neurons expected per patch
gSig    = [2,2,1]                   # expected half size of neurons
merge_thresh = 0.8                  # merging threshold, max correlation allowed
p       = 2                         # order of autoregressive system
gnb     = 2                         # number of global background components
init_method     = 'greedy_roi'      # initialisation method
alpha_snmf      = None              # this controls sparsity, e.g. 10e2
ssub    = 1                         # spatial subsampling during initialization
tsub    = 1                         # temporal subsampling during initialisation

# Make parameter object
#-------------------------------------------------------------------------------
opts_dict = {
    'fnames'        : tifs,
    'fr'            : fr,
    'decay_time'    : decay_time,
    'p'             : p,
    'nb'            : gnb,
    'rf'            : rf,
    'K'             : K,
    'rolling_sum'   : True,
    'only_init'     : True,
    'ssub'          : ssub,
    'tsub'          : tsub,
    'use_cnn'       : True,
}
opts = params.CNMFParams(params_dict = opts_dict)

# Play movie for illustration
#-------------------------------------------------------------------------------
if display_movie:
    m_orig = cm.load_movie_chain(tifs)

    ds_ratio = .2
    m_orig.resize(1,1,ds_ratio).play(q_max=99.5, fr = 30, magnification = 2)

# Memory Mapping
#-------------------------------------------------------------------------------
fname       = cm.save_memmap(tifs, base_name = 'Yr', is_3D = True, order = 'C')
Yr, dims, T = cm.load_memmap(fname)
images      = np.reshape(Yr.T, [T] + list(dims), order = 'F')       # load frames in python format (txy)

# Run CNMF on patches in parallel
#===============================================================================
opts.set('temporal', {'p':0})
cnm = cnmf.CNMF(n_processes, params = opts, dview = dview)
