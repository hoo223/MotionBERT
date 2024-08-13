import os, sys, getpass
user = getpass.getuser()
sys.path.append(f'/home/{user}/codes/hpe_library/')
from lib_import import *
from my_utils import *
os.chdir('/home/hrai/codes/MotionBERT')

from lib.data.datareader_total import DataReaderTotal
from lib.data.datareader_h36m import DataReaderH36M
from lib.data.datareader_fit3d import DataReaderFIT3D
from lib.utils.args import get_opts_args, get_opt_args_from_model_name
from lib.model.load_model import load_model
from lib.model.load_dataset import load_dataset
from lib.data.dataset_motion_3d import MotionDataset3DTotal
from lib.model.evaluation import *

blacklist_checkpoint = ['MB_train_h36m_gt_cam_no_factor_input_from_canonical_3d_same_z_s15678_tr_54138969_ts_others',
                        'MB_train_h36m_gt_cam_no_factor_input_from_canonical_3d_same_dist',
                        'MB_train_h36m_gt_cam_no_factor_input_from_canonical_3d_fixed_dist_tr_s1_ts_s5678',
                        'MB_train_h36m_gt_cam_no_factor_input_from_canonical_3d_same_dist_input_centering_tr_s1_ts_s5678',
                        'MB_train_h36m_gt_cam_no_factor_input_from_canonical_3d_same_dist_s15678_tr_54138969_ts_others',
                        'MB_train_h36m_gt_cam_no_factor_input_from_canonical_3d_same_dist_tr_s1_ts_s5678',
                        'MB_train_h36m_gt_cam_no_factor_input_from_canonical_3d_fixed_dist_5',
                        ]
experiment_list = [item.split('.')[0] for item in os.listdir('experiments') if item not in blacklist_checkpoint]

subset_list = [
    '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_FIXED_DIST_5_ADAPTIVE_FOCAL-TEST_ALL_TRAIN',
    '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_FIXED_DIST_5_ADAPTIVE_FOCAL-TEST_TS1_6',
    '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_FIXED_DIST_5-TEST_ALL_TRAIN',
    '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_FIXED_DIST_5-TEST_TS1_6',
    '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_Z-TEST_ALL_TRAIN',
    '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_Z-TEST_TS1_4',
    '3DHP-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_Z-TEST_TS1_6',
    '3DHP-GT-CAM_NO_FACTOR-POSEAUG_TEST_2929',
    '3DHP-GT-CAM_NO_FACTOR-POSYNDA_TESTSET',
    '3DHP-GT-CAM_NO_FACTOR-TEST_ALL_TRAIN',
    '3DHP-GT-CAM_NO_FACTOR-TEST_TS1_4',
    '3DHP-GT-CAM_NO_FACTOR-TEST_TS1_6',
    'FIT3D-GT-ALL_TEST',
    'FIT3D-GT-CAM_NO_FACTOR-TR_S03',
    'FIT3D-GT-CAM_NO_FACTOR-ALL_TEST',
    'FIT3D-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_FIXED_DIST_5_ADAPTIVE_FOCAL-ALL_TEST',
    'FIT3D-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_FIXED_DIST_5-ALL_TEST',
    'FIT3D-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_Z-ALL_TEST',
    'FIT3D-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_Z-TR_S03',
    'FIT3D-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_Z-TS_S4710',
    'FIT3D-GT-CAM_NO_FACTOR-TS_S4710',
    'FIT3D-GT-TS_S4710',
    'H36M-GT',
    'H36M-GT-CAM_NO_FACTOR',
    'H36M-GT-CAM_NO_FACTOR-TEST_ALL',
    'H36M-GT-CAM_NO_FACTOR-TR_S1',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_FIXED_DIST_5',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_FIXED_DIST_5_ADAPTIVE_FOCAL',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_FIXED_DIST_5-TR_S1_TS_S5678',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_FIXED_DIST_5_ADAPTIVE_FOCAL-TR_S1_TS_S5678',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_FIXED_DIST_5-TR_S1',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_FIXED_DIST_5_ADAPTIVE_FOCAL-TR_S1',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_DIST',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_DIST-S15678_TR_54138969_TS_OTHERS',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_DIST-TR_S1_TS_S5678',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_Z',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_Z-TEST_ALL',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_Z-TR_S1',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_Z-TR_S1_TS_S5678',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_Z-STEP_ROT_1-TR_S1_TS_S5678',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_Z-TR_S1_TS_S5678_BY_CANON1_6_PRED',
    'H36M-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_Z-TR_S1_TS_S5678_BY_CANON2_2_PRED',
    'H36M-GT-CAM_NO_FACTOR-S15678_TR_54138969_TS_OTHERS',
    'H36M-GT-CAM_NO_FACTOR-TR_S1_TS_S5678',
    'H36M-GT-INPUT_FROM_3D_CANONICAL_SAME_DIST-TR_S1_TS_S5678',
    'H36M-GT-TR_S1_TS_S5678',
    'H36M-GT-WORLD_NO_FACTOR',
    'H36M-SH',
    'H36M-CANONICALIZATION-GT-INPUT_FROM_3D_CANONICAL_SAME_Z-TR_S1_TS_S5678',
]