import os, sys
sys.path.append('/home/hrai/codes/hpe_library')
from hpe_library.lib_import import natsorted, plt, Rotation, radians, cos, sin, draw3d_arrow
from my_utils import get_pose_seq_and_cam_param, rotate_torso_by_R, axes_3d, axes_2d, load_h36m, get_rootrel_pose, World2CameraCoordinate, projection, normalize_screen_coordinates, draw_3d_pose, draw_2d_pose, clear_axes, rotation_matrix_from_vectors, Camera, T_to_C
from hpe_library.lib_import import *
from hpe_library. my_utils import *


import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual, interactive_output
import threading
from ipywidgets import GridspecLayout
from ipywidgets import TwoByTwoLayout

box_layout = widgets.Layout(
        border='solid 1px red',
        margin='0px 10px 10px 0px',
        padding='5px 5px 5px 5px')

os.chdir('/home/hrai/codes/MotionBERT/custom_codes/canonical/utils_camera_analysis')
from .button import *
from .text import *
from .slider import *
from .init_camera import *
from .select import *
from .load_dataset import *
from .init_interactive import *
from .init_panel import *
from .init_plot import *
from .init_layout import *
from .interactive_func import *
from .button_func import *
from .functions import *