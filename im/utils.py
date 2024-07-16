import os.path as osp

import numpy as np
from argparse import ArgumentParser
from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.lines import Lines

MESH_COLOR = (149 / 255, 85 / 255, 149 / 255, 0.5)
LINE_COLOR = (0 / 255, 85 / 255, 200 / 255, 0.5)
