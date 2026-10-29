"""
Channelmap Generator for Neuropixels Probes

A Python package for generating channel maps for Neuropixels probes with hardware constraints.
Includes both a programmatic API and an interactive browser-based GUI.
"""

from .backend import *
from .constants import PROBE_TYPE_MAP as PROBE_TYPE_MAP  # public API re-export

__version__ = "1.4.1"
__author__ = "Maxime Beau"
__email__ = "maxime@princeton.edu"
