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
from caiman.components_evaluation import evaluate_components, estimate_components_quality_auto
from caiman.cluster import setup_cluster
from caiman.paths import caiman_datadir

import bokeh.plotting as bpl

# Housekeeping
#===============================================================================
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
print('Stopping cluster to avoid unnecessary use of memory')
sys.stdout.flush()
cm.stop_server()

# Set Parameters (taken from standard demo)
#-------------------------------------------------------------------------------
rs      = (15,15,15)                # half-ize of patches in pixels
stride  = (10,10,10)                # amount of overlap between patches in pixels
K       = 12                        # number of neurons expected per patch
gSig    = [2,2,1]                   # expected half size of neurons
merge_thresh = 0.8                  # merging threshold, max correlation allowed
p       = 2                         # order of autoregressive system
save_results = False                # flag to save results or not
init_method     = 'greedy_roi'      # initialisation method
alpha_snmf      = None              # this controls sparsity, e.g. 10e2

# Run algorithm on patches
#-------------------------------------------------------------------------------
# cmm - cnmf.CNMF(n_processes)
