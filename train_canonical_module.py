import sys
sys.path.append('/home/hrai/codes/hpe_library')
from lib_import import *
from my_utils import *
os.chdir('/home/hrai/codes/MotionBERT')

import os
import numpy as np
import random
import torch

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

def main(args, opts):
    # Load dataset
    train_loader_3d, test_loader, posetrack_loader_2d, instav_loader_2d, datareader = load_dataset(args)
    # Load model and checkpoint
    model_pos, chk_filename, checkpoint = load_model(opts, args)
    # main process
    if not opts.evaluate: # Training
        train(args, opts, checkpoint, model_pos, train_loader_3d, posetrack_loader_2d, instav_loader_2d, test_loader, datareader)
    elif opts.evaluate: # Evaluation
        e1, e2, results_all, inputs_all, gts_all, total_result_dict = evaluate(args, model_pos, test_loader, datareader, checkpoint)

if __name__ == "__main__":
    args, opts = get_opts_args()
    set_random_seed(opts.seed)
    print(args)
    main(args, opts)