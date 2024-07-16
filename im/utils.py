import os.path as osp

import numpy as np
from argparse import ArgumentParser
from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLSequence

MODEL_COLOR = (149 / 255, 85 / 255, 149 / 255, 0.5)
