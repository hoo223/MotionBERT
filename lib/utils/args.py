import argparse
from lib.utils.tools import *

def list_of_strings(arg):
    return arg.split(',')

def get_opts_args(input_args=None, verbose=True):
    opts = parse_args(input_args)
    args = get_config(opts.config)
    args.seed = opts.seed
    # check arguments
    args, opts = check_args(args, opts)
    if verbose:
        print(opts.config)
        print(opts.pretrained_backbone == '')
    return args, opts

def get_opt_args_from_model_name(checkpoint, config_root = 'configs/pose3d/', checkpoint_root = 'checkpoint/pose3d/', mode='best', verbose=True):
    # mode: best or lastest
    config = checkpoint + '.yaml'
    if 'fit3d' in checkpoint:
        if 'ts_s4710' in checkpoint: checkpoint_root = 'checkpoint/pose3d_fit3d_ts_s4710/'
        elif 'tr_s03' in checkpoint: checkpoint_root = 'checkpoint/pose3d_fit3d_tr_s03/'
        else: raise ValueError(f'Unknown fit3d model {checkpoint}')
    elif 'h36m' in checkpoint:
        if 'tr_s1_ts_s5678' in checkpoint: checkpoint_root = 'checkpoint/pose3d_h36m_tr_s1_ts_s5678/'
        elif 'tr_s15_ts_s678' in checkpoint: checkpoint_root = 'checkpoint/pose3d_h36m_tr_s15_ts_s678/'
        else: checkpoint_root = 'checkpoint/pose3d_h36m/'
    elif '3dhp' in checkpoint:
        if 'tr_s1' in checkpoint: checkpoint_root = 'checkpoint/pose3d_3dhp_tr_s1/'
        else: checkpoint_root = 'checkpoint/pose3d_3dhp/'
    assert mode in ['best', 'latest'], 'mode should be best or lastest'
    bin_file = '/' + mode + '_epoch.bin'
    input_args = ['--config', config_root + config, '--evaluate', checkpoint_root + checkpoint + bin_file]
    args, opts = get_opts_args(input_args, verbose)
    return args, opts

def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-pb', '--pretrained_backbone', default='', type=str, metavar='PATH', help='pretrained backbone checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-g', '--gpu', default='0, 1', type=str, help='GPU id')
    parser.add_argument('--part_list', type=str, nargs='+', help='eval part list')
    parser.add_argument('-tr', '--test_run', action='store_true', help='test run')
    if type(input_args) == type(None):
        opts = parser.parse_args()
    else:
        opts = parser.parse_args(input_args)
    return opts

def check_args(args, opts):
    # model
    try: test = args.model
    except:
        if opts.evaluate: args.model = opts.evaluate.split('/')[2]
        else: args.model = 'MB'
    # part list
    if type(opts.part_list) != type(None):
        args.part_list = opts.part_list
    try: test = args.part_list
    except: args.part_list = ['whole']
    try: test = args.eval_part
    except:
        if 'whole' in args.part_list: args.eval_part = 'whole'
        else: args.eval_part = args.part_list[0]
    # loss weights
    try: test = args.lambda_3d_pos
    except: args.lambda_3d_pos = 0.0
    try: test = args.lambda_scale
    except: args.lambda_scale = 0.0
    try: test = args.lambda_3d_velocity
    except: args.lambda_3d_velocity = 0.0
    try: test = args.lambda_limb_pos
    except: args.lambda_limb_pos = 0.0
    try: test = args.lambda_limb_scale
    except: args.lambda_limb_scale = 0.0
    try: test = args.lambda_limb_velocity
    except: args.lambda_limb_velocity = 0.0
    try: test = args.lambda_torso_pos
    except: args.lambda_torso_pos = 0.0
    try: test = args.lambda_lower_frame_R
    except: args.lambda_lower_frame_R = 0.0
    try: test = args.lambda_upper_frame_R
    except: args.lambda_upper_frame_R = 0.0
    try: test = args.lambda_lv
    except: args.lambda_lv = 0.0
    try: test = args.lambda_lg
    except: args.lambda_lg = 0.0
    try: test = args.lambda_a
    except: args.lambda_a = 0.0
    try: test = args.lambda_av
    except: args.lambda_av = 0.0
    try: test = args.lambda_sym
    except: args.lambda_sym = 0.0
    try: test = args.lambda_root_point
    except: args.lambda_root_point = 0.0
    try: test = args.lambda_length
    except: args.lambda_length = 0.0
    try: test = args.lambda_dh_angle
    except: args.lambda_dh_angle = 0.0
    try: test = args.lambda_onevec_pos
    except: args.lambda_onevec_pos = 0.0
    try: test = args.lambda_dh_angle2
    except: args.lambda_dh_angle2 = 0.0
    try: test = args.lambda_dh_length
    except: args.lambda_dh_length = 0.0
    try: test = args.lambda_canonical_2d_residual
    except: args.lambda_canonical_2d_residual = 0.0
    try: test = args.canonical_loss
    except: args.canonical_loss = 'mpjpe'
    # input centering
    try: test = args.input_centering
    except: args.input_centering = False
    try: test = args.norm_input_scale
    except: args.norm_input_scale = False
    # finetune only head
    try: test = args.finetune_only_head
    except: args.finetune_only_head = False
    # freeze backbone
    try: test = args.freeze_backbone
    except: args.freeze_backbone = False
    # calculate mpjpe after part
    try: test = args.mpjpe_after_part
    except: args.mpjpe_after_part = False
    # default input mode
    try: test = args.input_mode
    except: args.input_mode = 'joint_2d' # joint_2d_from_canonical_3d
    # default gt 3d mode
    try: test = args.gt_mode
    except: args.gt_mode = 'joint3d_image' # joint3d_image, cam_3d, world_3d
    try: test = args.mpjpe_mode
    except:
        if args.gt_mode == 'joint3d_image': args.mpjpe_mode = 'joints_2.5d_image'
        else: args.mpjpe_mode = args.gt_mode
    # default steprot
    #try: test = args.step_rot
    #except: args.step_rot = 0
    # denormalize oupput of the model (3d pose)
    try: test = args.denormalize_output
    except:
        args.denormalize_output = True
        if args.gt_mode == 'cam_3d': args.denormalize_output = False
    # print summary table
    try: test = args.print_summary_table
    except: args.print_summary_table = False
    # model input, output dim
    try: test = args.dim_in
    except: args.dim_in = 3
    try: test = args.dim_out
    except: args.dim_out = 3
    # fix orientation
    try: test = args.fix_orientation_pred
    except: args.fix_orientation_pred = False
    try: test = args.fix_orientation_gt
    except: args.fix_orientation_gt = False
    # default data type list
    try: test = args.default_data_type_list
    except: args.default_data_type_list = ['source', 'cam_param', 'camera_name', 'action', 'confidence']
    # normalize 2d
    try: test = args.normalize_2d
    except: args.normalize_2d = True
    # use_new_datareader
    try: test = args.use_new_datareader
    except: args.use_new_datareader = True

    return args, opts
