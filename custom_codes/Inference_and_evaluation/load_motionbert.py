import os
import torch
import random
import argparse
import sys
import numpy as np

sys.path.append('/home/hrai/codes/MotionBERT')
from lib.utils.learning import *
from lib.utils.tools import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/hrai/codes/MotionBERT/configs/pose3d/MB_ft_h36m.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='/home/hrai/codes/MotionBERT/checkpoint/pose3d/MB_ft_h36m/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    #parser.add_argument('-ms', '--selection', default='', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args([])
    return opts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_motionbert():

    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)

    model_backbone = load_backbone(args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    print('Loading checkpoint', opts.evaluate)
    checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
    model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    model_pos = model_backbone
    model_pos.eval()

    return model_pos