import logging
import matplotlib as plt
import numpy as np
import os
import psutil
from scipy.ndimage.filters import gaussian_filter
import sys

import caiman as cm
from caiman.utils.visualization import nb_view_patches3d
import caiman.source_extraction.cnmf as cnmf
from caiman.components_evaluation import evaluate_components, estimate_components_quality_auto
from caiman.cluster import setup_cluster
from caiman,oaths import caiman_datadir

import bokeh.plotting as bpl
