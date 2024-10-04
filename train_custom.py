import sys
sys.path.append('/home/hrai/codes/hpe_library')
from hpe_library.lib_import import *
from hpe_library. my_utils import *
os.chdir('/home/hrai/codes/MotionBERT')

import os
import numpy as np
import random
import torch
#import wandb
#wandb.login()


from lib.utils.learning import * # load_backbone
from lib.utils.args import get_opts_args
from lib.utils.utils_data import flip_data
from lib.model.load_dataset import load_dataset
from lib.model.load_model import load_model
from lib.model.loss import *
from lib.model.training import *
from lib.model.evaluation import *

os.environ["NCCL_P2P_DISABLE"]= '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def get_project_name(args):
    item = args.model
    project_name = ''
    if 'CanonDSTformer' in item:
        project_name += 'Canonicalization_Network_'
    if 'h36m' in item:
        project_name += 'H36M'
        if '_gt' in item:
            project_name += '_GT'
        else:
            project_name += '-SH'
    elif 'fit3d' in item:
        project_name += 'FIT3D'
    elif 'kookmin' in item:
        project_name += 'KOOKMIN'
        
    if '_ts_s4710' in item:
        project_name += '-TS_S4710'
    elif '_tr_s1_ts_s5678' in item:
        project_name += '-TR_S1_TS_S5678'
    elif '_s15678_tr_54138969_ts_others' in item:
        project_name += '-S15678_TR_54138969_TS_OTHERS'
    
    return project_name

def main(args, opts, run=None):
    # Load dataset
    train_loader_3d, test_loader, posetrack_loader_2d, instav_loader_2d, datareader = load_dataset(args, use_new_datareader=True)
    # Load model and checkpoint
    model_pos, chk_filename, checkpoint = load_model(opts, args)
    # main process
    if not opts.evaluate: # Training
        train(args, opts, checkpoint, model_pos, train_loader_3d, posetrack_loader_2d, instav_loader_2d, test_loader, datareader, run)
    elif opts.evaluate: # Evaluation
        e1, e2, results_all, inputs_all, gts_all, total_result_dict = evaluate(args, model_pos, test_loader, datareader, checkpoint)

if __name__ == "__main__":
    args, opts = get_opts_args()
    set_random_seed(opts.seed)
    print(args)
    # if opts.evaluate == '':
    #     # run = wandb.init(
    #     #     project=get_project_name(args),
    #     #     config=args,
    #     #     name=args.model,
    #     # )
    #     run = None
    # else:
    #     run = None
    main(args, opts)
    # if run != None:
    #     wandb.finish()
    
    