from time import time
import os.path as osp
import multiprocessing as mp
from typing import List, Dict, Optional, Any, Tuple, Set

import numpy as np
from scipy.spatial import Delaunay
from argparse import ArgumentParser

from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.lines import Lines

MESH_COLOR = (149 / 255, 85 / 255, 149 / 255, 0.5)
LINE_COLOR = (0 / 255, 85 / 255, 200 / 255, 0.5)


def all_larger_than(s: set, val: int) -> bool:
    return all(x > val for x in s)


def all_smaller_than(s: set, val: int) -> bool:
    return all(x < val for x in s)
