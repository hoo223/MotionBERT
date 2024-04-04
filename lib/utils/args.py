import argparse

def list_of_strings(arg):
    return arg.split(',')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-g', '--gpu', default='0, 1', type=str, help='GPU id')
    parser.add_argument('--part_list', type=str, nargs='+', help='eval part list')
    parser.add_argument('-tr', '--test_run', action='store_true', help='test run')
    opts = parser.parse_args()
    return opts

def check_args(args, opts):
    try: test = args.model
    except:
        if opts.evaluate: args.model = opts.evaluate.split('/')[2]
        else: args.model = 'MB'
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

    try: test = args.canonical
    except: args.canonical = False
    
    try: test = args.finetune_only_head
    except: args.finetune_only_head = False

    try: test = args.mpjpe_after_part
    except: args.mpjpe_after_part = False

    try: test = args.gt_mode
    except: args.gt_mode = 'joint3d_image'

    try: test = args.denormalize_output
    except: args.denormalize_output = True

    return args