################################################################################
## Code adapted from demo_caiman_cnmf_3D as imported from github 21/11/2018
## https://github.com/flatironinstitute/CaImAn
################################################################################

# Import relevant packages
#===============================================================================
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import shutil
from scipy.ndimage.filters import gaussian_filter
import scipy.sparse
import sys
import re
from skimage.external.tifffile import imread
from skimage import io

# Caiman setup
#-------------------------------------------------------------------------------
import caiman as cm
import caiman.source_extraction.cnmf as cnmf
from caiman.utils.visualization import nb_view_patches3d
from caiman.source_extraction.cnmf import params as params
from caiman.components_evaluation import evaluate_components, estimate_components_quality_auto
from caiman.motion_correction import MotionCorrect
from caiman.cluster import setup_cluster
from caiman.paths import caiman_datadir
from caiman.utils.visualization import inspect_correlation_pnr

# Housekeeping
#===============================================================================
# Module flags
#-------------------------------------------------------------------------------
display_movie   = False      # play movie of tifs that are loaded
save_results    = False      # flag to save results or not
mot_correct     = True
pid             = 7

# Define folder locations
#-------------------------------------------------------------------------------
Fbase  = '/Volumes/MARIANNE'
Fmvct  = Fbase
Fshort = Fmvct + os.sep + 'Short_BL'
Fbln   = Fbase + os.sep + 'BL'

# Load tifs
#-------------------------------------------------------------------------------
Fthis   = Fbln                                          # Define which folder to use for the analysis

tifs    = []
ftifs   = os.listdir(Fthis)
print("I found " + str(len(ftifs)) + " files in total")
r       = re.compile('^[a-zA-Z0-9].*[tif|tiff]$')       # Regexp for relevant tifs
ftifs   = list(filter(r.match, ftifs))                  # matching tifs only
for ff in ftifs: tifs.append(os.path.join(Fthis, ff))   # reconstitute full file path
print("I found " + str(len(tifs)) + " tif files")


# Load specific plane from tif data - horribly convoluted way, sorry world
#===============================================================================
print('Working on plane number ' + str(pid))
use_existing = False

# Make plane for specific folder
#-------------------------------------------------------------------------------
Fplane = Fthis + os.sep + str(pid)
if os.path.isdir(Fplane):
    if not use_existing:
        shutil.rmtree(Fplane)
        os.mkdir(Fplane)

        # Save plane tifs in folder alone
        #-------------------------------------------------------------------------------
        cnt    = 0
        tst    = imread(tifs[0])[pid,:,:]
        pln    = np.empty((len(tifs), tst.shape[0], tst.shape[1]))
        pln    = pln.astype('uint16')

        for t in tifs:
            tpl           = np.nan_to_num(imread(t)[pid,:,:])
            pln[cnt,:,:]  = tpl.astype('uint16')
            cnt           = cnt + 1

        io.imsave(Fplane + os.sep + 'combined_'+ str(pid).zfill(3) + '.tif', pln)

    else:
        print("Beware: Using an existing folder, do double check")
else:
    os.mkdir(Fplane)

    # Save plane tifs in folder alone
    #-------------------------------------------------------------------------------
    cnt    = 0
    tst    = imread(tifs[0])[pid,:,:]
    pln    = np.empty((len(tifs), tst.shape[0], tst.shape[1]))
    pln    = pln.astype('uint16')

    for t in tifs:
        tpl           = np.nan_to_num(imread(t)[pid,:,:])
        pln[cnt,:,:]  = tpl.astype('uint16')
        cnt           = cnt + 1

    io.imsave(Fplane + os.sep + 'combined_'+ str(pid).zfill(3) + '.tif', pln)


# Get list of all tifs from this plane
#-------------------------------------------------------------------------------
ptifs = os.listdir(Fplane)
cnt   = 0
tf    = []
r     = re.compile('[a-z]+.*[0-9]*.*[tif|tiff]$')
ptifs = list(filter(r.match, ptifs))

for pp in ptifs:
    tf.append(Fplane + os.sep + pp)
    cnt = cnt + 1

print('I am using ' + str(len(tf)) + ' file(s)')
tf

# Segmentation parameter settings
#===============================================================================
# Start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

# Set Parameters (taken from standard demo)
#-------------------------------------------------------------------------------
# Dataset dependent parameters
#-------------------------------------------------------------------------------
fr          = 4                     # imaging rate in frames per seconds
decay_time  = .4                    # length of a typical transient in seconds

# Motion correction parameters
#-------------------------------------------------------------------------------
strides = (96, 96)          # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)         # overlap between pathes (size of patch strides+overlaps)
max_shifts = (6,6)          # maximum allowed rigid shifts (in pixels)
max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
pw_rigid = True             # flag for performing non-rigid motion correction


# Parameters for source extraction and deconvolution
#-------------------------------------------------------------------------------
p = 1                       # order of the autoregressive system
gnb = 0                     # number of global background components
nb_patch = 0                # number of bg comps per patch if gnb > 0
merge_thresh = 0.7          # merging threshold, max correlation allowed
rf = 40                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 10            # am ount of overlap between the patches in pixels
K = None                    # number of components per patch
gSig = (.5,.5)              # expected half size of neurons in pixels
gSiz = (2,2)
Ain  = None                 # Can seed with pre-determined binary mask here
method_init = 'corr_pnr'    # initialization method (if analyzing dendritic data using 'sparse_nmf')
ssub = 1                    # spatial subsampling during initialization
tsub = 3                    # temporal subsampling during intialization
low_rank_background = None  # None leaves backgroiund of each patch intact, true performs global low-rank approx if gnb > 0
min_corr = 0                # min peak value from correlation image
min_pnr = 5                 # min peak to bnoise ration from PNR image
ssub_B = 2                  # additional downsampling factor in space for backgrouund
ring_size_factor = 1.2      # radius of ring is gSiz * ring_size_factor

# Parameters for component evaluation
#-------------------------------------------------------------------------------
min_SNR = 2.0               # signal to noise ratio for accepting a component
rval_thr = 0.85             # space correlation threshold for accepting a component
cnn_thr = 0.99              # threshold for CNN based classifier
cnn_lowest = 0.1            # neurons with cnn probability lower than this value are rejected

# Create parameters dictionary
#-------------------------------------------------------------------------------
opts_dict = {'method_init': method_init,
            'K': K,
            'gSig' : gSig,
            'gSiz' : gSiz,
            'merge_thresh': merge_thresh,
            'p': p,
            'ssub': ssub,
            'tsub': tsub,
            'rf': rf,
            'strides': strides,
            'only_init': True,
            'nb': gnb,
            'nb_patch': nb_patch,
            'method_deconvolution': 'oasis',
            'low_rank_background': low_rank_background,
            'update_background_components': True,
            'min_corr': min_corr,
            'min_pnr': min_pnr,
            'normalize_init': False,
            'center_psf': True,
            'ssub_B': ssub_B,
            'ring_size_factor': ring_size_factor,
            'del_duplicates': True,
            'fnames': tf,
            'decay_time': decay_time,
            'border_pix': 10,
            'overlaps': overlaps,
            'max_shifts': max_shifts,
            'max_deviation_rigid': max_deviation_rigid,
            'pw_rigid': pw_rigid,
            'stride': stride_cnmf,
            'rolling_sum': True,
            'min_SNR': min_SNR,
            'rval_thr': rval_thr,
            'use_cnn': True,
            'min_cnn_thr': cnn_thr,
            'cnn_lowest': cnn_lowest}

opts = params.CNMFParams(params_dict=opts_dict)

if mot_correct:
    # Motion correction setup
    #-------------------------------------------------------------------------------
    mc = MotionCorrect(tf, dview=None, **opts.get_group('motion'))

    # Run piecewise-rugu nituib cirrectuib ysubg NoRMCorre
    #-------------------------------------------------------------------------------
    mc.motion_correct(save_movie=True)
    m_els = cm.load(mc.fname_tot_els)
    border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0       # maximum shift to be used for trimming against NaNs

    # Compare wth original movie - This isn't currently working particularly well
    #-------------------------------------------------------------------------------
    if display_movie:
        m_orig = cm.load_movie_chain(tf)
        ds_ratio = 0.2
        cm.concatenate([m_orig.resize(1,1,ds_ratio) - mc.min_mov*mc.nonneg_movie,
                        m_els.resize(1,1,ds_ratio)],
                        axis = 2).play(fr=60, gain=15, magnification=2, offset=0) # press q to exit

# Memory Mapping
#-------------------------------------------------------------------------------
mapfile     = cm.save_memmap(mc.mmap_file, base_name = 'memmap_', order = 'C')
Yr, dims, T = cm.load_memmap(mapfile)
images      = np.reshape(Yr.T, [T] + list(dims), order = 'F')

# compute some summary images (correlation and peak to noise)
#-------------------------------------------------------------------------------
cn_filter, pnr = cm.summary_images.correlation_pnr(images, gSig=gSig[0], swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
# inspect the summary images and set the parameters
inspect_correlation_pnr(cn_filter, pnr)
print(min_corr)
print(min_pnr)

# Restart cluster to clean up memory
#-------------------------------------------------------------------------------
cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

#===============================================================================
# Run the actual Segmentation
#===============================================================================
cnm = cnmf.CNMF(n_processes, params=opts, dview=dview, Ain=Ain)
cnm = cnm.fit(images)


# Plot contours of found components
#-------------------------------------------------------------------------------
Cn = cm.local_correlations(images.transpose(1,2,0))
Cn[np.isnan(Cn)] = 0
cnm.estimates.plot_contours_nb(img=Cn)

#%% COMPONENT EVALUATION
#-------------------------------------------------------------------------------
# the components are evaluated in three ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
#   c) each shape passes a CNN based classifier - this doesn't currently work in this implementation

min_SNR = 3            # adaptive way to set threshold on the transient size
r_values_min = 0.85    # threshold on space consistency (if you lower more components
#                        will be accepted, potentially with worst quality)
cnm.params.set('quality', {'min_SNR': min_SNR,
                           'rval_thr': r_values_min,
                           'use_cnn': False})
cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

print(' ***** ')
print('Number of total components: ', len(cnm.estimates.C))
print('Number of accepted components: ', len(cnm.estimates.idx_components))

#%% PLOT COMPONENTS
#-------------------------------------------------------------------------------
cnm.estimates.plot_contours_nb(img=Cn, idx=cnm.estimates.idx_components)

# Save some estimates or other 
#-------------------------------------------------------------------------------
fsave = Fplane + os.sep + 'compressed_cell_idx.npz'
if os.path.isfile(fsave):
    print('Deleting old results')
    os.remove(fsave)

scipy.sparse.save_npz(Fplane + os.sep + 'compressed_cell_idx.npz', cnm.estimates.A)
