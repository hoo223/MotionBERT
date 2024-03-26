import re
import sys
sys.path.append('/home/hrai/codes/hpe_library')
from lib_import import *
from my_utils import *
os.chdir('/home/hrai/codes/MotionBERT')

import os
import numpy as np
import argparse
import ast
import errno
import math
import pickle
import tensorboardX
from tqdm import tqdm
from time import time
import copy
import random
import prettytable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.tools import * # get_config
from lib.utils.learning import * # load_backbone
from lib.utils.utils_data import flip_data
from lib.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.data.augmentation import Augmenter2D
from lib.data.datareader_h36m import DataReaderH36M
from lib.data.datareader_aihub import DataReaderAIHUB
from lib.data.datareader_fit3d import DataReaderFIT3D
from lib.data.datareader_kookmin import DataReaderKOOKMIN
from lib.model.loss import *
from lib.model.DHDSTformer import DHDSTformer_total, DHDSTformer_total2, DHDSTformer_total3, \
    DHDSTformer_limb, DHDSTformer_limb2, DHDSTformer_limb3, DHDSTformer_limb4, DHDSTformer_limb5, \
    DHDSTformer_right_arm, DHDSTformer_right_arm2, DHDSTformer_right_arm3, \
    DHDSTformer_torso, DHDSTformer_torso2, \
    DHDSTformer_torso_limb, \
    DHDSTformer_onevec

os.environ["NCCL_P2P_DISABLE"]= '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

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

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss' : min_loss
    }, chk_path)
    
def evaluate_onevec(args, model_pos, test_loader, datareader):
    gts_all_root_point = []
    gts_all_child_point = []
    gts_all_3d_pos = []
    gts_all_length = []
    gts_all_angle = []
    gts_all_child_point = []
    results_all_root_point = []
    results_all_child_point = []
    results_all_3d_pos = []
    results_all_length = []
    results_all_angle = []
    results_all_child_point = []
    model_pos.eval()           
    num_test_frames = 0 
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader): # batch_input: normalized joint_2d, batch_gt: normalized joint3d
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()    
                batch_gt = batch_gt.cuda()
            # inference
            input, gt_root_point, gt_length, gt_angle = get_input_gt_for_onevec(batch_input, batch_gt)
            pred_root_point, pred_length, pred_angle = model_pos(input)
            
            num_test_frames += input.shape[0] * input.shape[1]
            
            gt_R_azim = gt_angle[:, :, 0] # torch.Size([B, F])
            gt_R_elev = gt_angle[:, :, 1] # torch.Size([B, F])
            pred_R_azim = pred_angle[:, :, 0] # torch.Size([B, F])
            pred_R_elev = pred_angle[:, :, 1] # torch.Size([B, F])
            gt_child_point = batch_azim_elev_to_vec(gt_R_azim, gt_R_elev, gt_length, gt_root_point) # torch.Size([B, F, 3])
            pred_child_point = batch_azim_elev_to_vec(pred_R_azim, pred_R_elev, pred_length, pred_root_point) # torch.Size([B, F, 3])
            gt_3d_pos = torch.cat([gt_root_point.unsqueeze(2), gt_child_point.unsqueeze(2)], dim=2) # torch.Size([B, F, 2, 3])
            pred_3d_pos = torch.cat([pred_root_point.unsqueeze(2), pred_child_point.unsqueeze(2)], dim=2) # torch.Size([B, F, 2, 3])
            
            if args.rootrel:
                gt_3d_pos = gt_root_point - gt_3d_pos[:, :, 0:1, :]
            else:
                gt_3d_pos[:, :, :, 2] = gt_3d_pos[:, :, :, 2] - gt_3d_pos[:, 0:1, 0:1, 2] 
            
            results_all_root_point.append(pred_root_point.cpu().numpy())
            results_all_child_point.append(pred_child_point.cpu().numpy())
            results_all_3d_pos.append(pred_3d_pos.cpu().numpy())
            results_all_length.append(pred_length.cpu().numpy())
            results_all_angle.append(pred_angle.cpu().numpy())
            gts_all_root_point.append(gt_root_point.cpu().numpy())
            gts_all_child_point.append(gt_child_point.cpu().numpy())
            gts_all_3d_pos.append(gt_3d_pos.cpu().numpy())
            gts_all_length.append(gt_length.cpu().numpy())
            gts_all_angle.append(gt_angle.cpu().numpy())
            
    results_all_root_point = np.concatenate(results_all_root_point)
    results_all_child_point = np.concatenate(results_all_child_point)
    results_all_3d_pos = np.concatenate(results_all_3d_pos)
    results_all_length = np.concatenate(results_all_length)
    results_all_angle = np.concatenate(results_all_angle)
    gts_all_root_point = np.concatenate(gts_all_root_point)
    gts_all_child_point = np.concatenate(gts_all_child_point)
    gts_all_3d_pos = np.concatenate(gts_all_3d_pos)
    gts_all_length = np.concatenate(gts_all_length)
    gts_all_angle = np.concatenate(gts_all_angle)
    
    # print(f"Root point: {results_all_root_point.shape}, {gts_all_root_point.shape}")
    # print(f"Length: {results_all_length.shape}, {gts_all_length.shape}")
    # print(f"Angle: {results_all_angle.shape}, {gts_all_angle.shape}")
    
    num_test_frames = int(num_test_frames/4)
    
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    part_list = ['whole', 'r_hip', 'l_hip', 'r_knee', 'l_knee', 'r_shoulder', 'l_shoulder', 'r_elbow', 'l_elbow', 'r_upper_leg', 'l_upper_leg', 'r_upper_arm', 'l_upper_arm']
    eval_part = 'whole'

    total_result_dict = {}
    for part in part_list:
        total_result_dict[part] = {
            'e1_all': np.zeros(num_test_frames),
            'e2_all': np.zeros(num_test_frames),
            'oc': np.zeros(num_test_frames),
            'results': {},
            'results_procrustes': {}
        }
        
        # To classify the results by action
        for action in action_names:
            total_result_dict[part]['results'][action] = []
            total_result_dict[part]['results_procrustes'][action] = []
    
    for idx in range(len(results_all_3d_pos)):
        pred_root_point = results_all_root_point[idx] # torch.Size([F, 3])
        pred_child_point = results_all_child_point[idx] # torch.Size([F, 3])
        pred_3d_pos = results_all_3d_pos[idx] # torch.Size([F, 2, 3])
        pred_length = results_all_length[idx] # torch.Size([1])
        pred_angle = results_all_angle[idx] # torch.Size([F, 2])
        gt_root_point = gts_all_root_point[idx] # torch.Size([F, 3])
        gt_child_point = gts_all_child_point[idx] # torch.Size([F, 3])
        gt_3d_pos = gts_all_3d_pos[idx] # torch.Size([F, 2, 3])
        gt_length = gts_all_length[idx] # torch.Size([1])
        gt_angle = gts_all_angle[idx] # torch.Size([F, 2])
        
        err1_per_joint = mpjpe_for_each_joint(gt_3d_pos* 1000, pred_3d_pos* 1000)  # (F, J)
        err2_per_joint = p_mpjpe_for_each_joint(gt_3d_pos* 1000, pred_3d_pos* 1000) # (F, J)
        
        if idx%4 == 0: part = 'r_upper_leg'
        elif idx%4 == 1: part = 'l_upper_leg'
        elif idx%4 == 2: part = 'r_upper_arm'
        elif idx%4 == 3: part = 'l_upper_arm'
        
        frame_list = list(range((idx//4)*args.clip_len, ((idx//4)+1)*args.clip_len))
        total_result_dict[part]['e1_all'][frame_list] = np.mean(err1_per_joint, axis=1) # (243, ) # 각 프레임 별 에러를 평균
        total_result_dict[part]['e2_all'][frame_list] = np.mean(err2_per_joint, axis=1) # (243, ) # 각 프레임 별 에러를 평균
        total_result_dict[part]['oc'][frame_list] = 1
        
        for J in range(err1_per_joint.shape[1]):
            if part == 'r_upper_leg':
                if J == 0: subpart = 'r_hip'
                elif J == 1: subpart = 'r_knee'
            elif part == 'l_upper_leg':
                if J == 0: subpart = 'l_hip'
                elif J == 1: subpart = 'l_knee'
            elif part == 'r_upper_arm':
                if J == 0: subpart = 'r_shoulder'
                elif J == 1: subpart = 'r_elbow'
            elif part == 'l_upper_arm':
                if J == 0: subpart = 'l_shoulder'
                elif J == 1: subpart = 'l_elbow'
            
            total_result_dict[subpart]['e1_all'][frame_list] += err1_per_joint[:, J] # (243, ) # 각 프레임 별 에러를 더해줌
            total_result_dict[subpart]['e2_all'][frame_list] += err2_per_joint[:, J] # (243, ) # 각 프레임 별 에러를 더해줌
            total_result_dict[subpart]['oc'][frame_list] += 1
            
            total_result_dict['whole']['e1_all'][frame_list] += err1_per_joint[:, J] # (243, ) # 각 프레임 별 에러를 더해줌
            total_result_dict['whole']['e2_all'][frame_list] += err2_per_joint[:, J] # (243, ) # 각 프레임 별 에러를 더해줌
            total_result_dict['whole']['oc'][frame_list] += 1
                    
    # Error per action
    actions = np.array(datareader.dt_dataset['test']['action'])
    for idx in tqdm(range(num_test_frames)):
        for part in part_list:
            if total_result_dict[part]['e1_all'][idx] > 0:
                err1 = total_result_dict[part]['e1_all'][idx] / total_result_dict[part]['oc'][idx]
                err2 = total_result_dict[part]['e2_all'][idx] / total_result_dict[part]['oc'][idx]
                action = actions[idx]
                total_result_dict[part]['results'][action].append(err1)
                total_result_dict[part]['results_procrustes'][action].append(err2)

    for part in ['whole']:
        print('Part:', part)
        final_result = []
        final_result_procrustes = []
        summary_table = prettytable.PrettyTable()
        summary_table.field_names = ['test_name'] + action_names # first row
        for action in action_names:
            final_result.append(np.mean(total_result_dict[part]['results'][action]))
            final_result_procrustes.append(np.mean(total_result_dict[part]['results_procrustes'][action]))
        summary_table.add_row(['P1 ({})'.format(part)] + final_result) # second row
        summary_table.add_row(['P2 ({})'.format(part)] + final_result_procrustes) # third row
        print(summary_table)
        
        # Total Error
        e1_part = np.mean(np.array(final_result))
        e2_part = np.mean(np.array(final_result_procrustes))
        print('Protocol #1 Error (MPJPE):', e1_part, 'mm')
        print('Protocol #2 Error (P-MPJPE):', e2_part, 'mm')
        print('----------------------------------------')
        if part == 'whole':
            e1 = e1_part
            e2 = e2_part
    return e1, e2, results_all_3d_pos, total_result_dict
    
def evaluate(args, model_pos, test_loader, datareader):
    print('INFO: Testing')
    results_all = []
    model_pos.eval()            
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader): # batch_input: normalized joint_2d, batch_gt: normalized joint3d_image
            N, T = batch_gt.shape[:2] # batch_size, 243
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.canonical:
                batch_input = batch_input - batch_input[:, :, 0:1, :] # root-relative
                batch_gt = batch_gt - batch_gt[:, :, 0:1, :]
            if batch_gt.shape[2] == 17:
                batch_gt_torso = batch_gt[:, :, [0, 1, 4, 7, 8, 9, 10, 11, 14], :] 
                batch_gt_limb_pos = batch_gt[:, :, [2, 3, 5, 6, 12, 13, 15, 16], :]
            # inference
            if args.flip:    
                batch_input_flip = flip_data(batch_input)
                if 'DHDSTformer_total' in args.model:
                    predicted_3d_pos_1 = model_pos(batch_input, length_type=args.test_length_type, ref_frame=args.length_frame)
                    predicted_3d_pos_flip = model_pos(batch_input_flip, length_type=args.test_length_type, ref_frame=args.length_frame)
                    predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)                   # Flip back
                    predicted_3d_pos = (predicted_3d_pos_1+predicted_3d_pos_2) / 2
                elif 'DHDSTformer_limb' in args.model:
                    predicted_3d_pos_1 = model_pos(batch_input, batch_gt_torso)
                    predicted_3d_pos_flip = model_pos(batch_input_flip, batch_gt_torso)
                    predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)                   # Flip back
                    predicted_3d_pos = (predicted_3d_pos_1+predicted_3d_pos_2) / 2
                elif 'DHDSTformer_torso' in args.model:
                    pred_torso_1, pred_lower_frame_R1, pred_upper_frame_R1 = model_pos(batch_input)
                    pred_torso_flip, pred_lower_frame_R_flip, pred_upper_frame_R_flip = model_pos(batch_input_flip)
                    pred_torso_2, pred_lower_frame_R2, pred_upper_frame_R2 = flip_data(pred_torso_flip)                   # Flip back
                    pred_torso = (pred_torso_1+pred_torso_2) / 2
                elif 'DHDSTformer_torso2' in args.model:
                    pred_torso_1 = model_pos(batch_input)
                    pred_torso_flip = model_pos(batch_input_flip)
                    pred_torso_2 = flip_data(pred_torso_flip)
                    pred_torso = (pred_torso_1+pred_torso_2) / 2
                else:
                    predicted_3d_pos_1 = model_pos(batch_input)
                    predicted_3d_pos_flip = model_pos(batch_input_flip)
                    predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)                   # Flip back
                    predicted_3d_pos = (predicted_3d_pos_1+predicted_3d_pos_2) / 2
            else:
                if 'DHDSTformer_total' in args.model:
                    predicted_3d_pos = model_pos(batch_input, length_type=args.test_length_type, ref_frame=args.length_frame)
                elif 'DHDSTformer_limb' in args.model:
                    predicted_3d_pos = model_pos(batch_input, batch_gt_torso)
                elif 'DHDSTformer_torso' in args.model:
                    pred_torso, pred_lower_frame_R, pred_upper_frame_R = model_pos(batch_input)
                    predicted_3d_pos = pred_torso
                elif 'DHDSTformer_torso2' in args.model:
                    pred_torso = model_pos(batch_input)
                    predicted_3d_pos = pred_torso
                else:
                    predicted_3d_pos = model_pos(batch_input)
            
            if args.rootrel:
                predicted_3d_pos[:,:,0,:] = 0     # [N,T,17,3]
            else:
                batch_gt[:,0,0,2] = 0
            if args.gt_2d: # input 2d를 추론값으로 사용함으로써 depth만 추정하도록 함
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())
    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all) # denormalize the predicted 3D poses

    _, split_id_test = datareader.get_split_id() # [range(0, 243) ... range(102759, 103002)] 
    actions = np.array(datareader.dt_dataset['test']['action']) # 103130 ['squat' ...  'kneeup']
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor']) # 103130 [3.49990559 ... 2.09230852]
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image']) # 103130, 17, 3
    sources = np.array(datareader.dt_dataset['test']['source']) # 103130 ['S02_6_squat_001' ... 'S08_4_kneeup_001']

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = np.array([actions[split_id_test[i]] for i in range(len(split_id_test))]) # actions[split_id_test]
    factor_clips = np.array([factors[split_id_test[i]] for i in range(len(split_id_test))]) # factors[split_id_test]
    source_clips = np.array([sources[split_id_test[i]] for i in range(len(split_id_test))]) # sources[split_id_test]
    frame_clips  = np.array([frames[split_id_test[i]] for i in range(len(split_id_test))]) # frames[split_id_test]
    gt_clips     = np.array([gts[split_id_test[i]] for i in range(len(split_id_test))]) # gts[split_id_test]
    assert len(results_all)==len(action_clips)

    total_result_dict = {}
    pelvis, r_hip, l_hip, torso, neck, l_shoulder, r_shoulder = 0, 1, 4, 7, 8, 11, 14
    r_knee, r_ankle, l_knee, l_ankle = 2, 3, 5, 6
    l_elbow, l_wrist, r_elbow, r_wrist = 12, 13, 15, 16
    nose, head = 9, 10
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    if 'H36M-SH' in args.subset_list:
        block_list = ['s_09_act_05_subact_02', 
                    's_09_act_10_subact_02', 
                    's_09_act_13_subact_01']
    else: block_list = []
            
    try:
        joint_list = args.eval_keypoint # Use only the specified keypoint number in config
        part = str(args.eval_keypoint)
        part_list = [part]
        args.eval_part = part
        total_result_dict[part] = {
            'e1_all': np.zeros(num_test_frames),
            'e2_all': np.zeros(num_test_frames),
            'oc': np.zeros(num_test_frames),
            'results': {},
            'results_procrustes': {}
        }

        # To classify the results by action
        for action in action_names:
            total_result_dict[part]['results'][action] = []
            total_result_dict[part]['results_procrustes'][action] = []
    except:
        print('No eval_keypoint. Use part list')
        part_list = args.part_list
        for part in part_list:
            total_result_dict[part] = {
                'e1_all': np.zeros(num_test_frames),
                'e2_all': np.zeros(num_test_frames),
                'oc': np.zeros(num_test_frames),
                'results': {},
                'results_procrustes': {}
            }
            
            # To classify the results by action
            for action in action_names:
                total_result_dict[part]['results'][action] = []
                total_result_dict[part]['results_procrustes'][action] = []

    for idx in range(len(results_all)):
        # check if the clip is in the block list
        if 'H36M-SH' in args.subset_list: 
            source = source_clips[idx][0][:-6]
        else: 
            source = source_clips[idx]
        if source in block_list:
            continue
        frame_list = frame_clips[idx] # range(idx*243, (idx+1)*243)
        action = action_clips[idx][0]
        factor = factor_clips[idx][:,None,None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        if 'no_factor' not in args.model:
            pred *= factor # scaling image to world scale
        
        # Root-relative Errors
        if args.model in ['MB', 'DHDSTformer_total', 'DHDSTformer_total2', 'DHDSTformer_total3', 'DHDSTformer_torso', 'DHDSTformer_torso_limb', 'DHDSTformer_right_upper_arm2']: # only model that predict pelvis point
            pred = pred - pred[:,0:1,:] # (243, 17, 3)
            gt = gt - gt[:,0:1,:] # (243, 17, 3)
        
        if 'DHDSTformer_torso' in args.model:
            gt = gt[:, [0, 1, 4, 7, 8, 9, 10, 11, 14], :] # (243, 9, 3) only torso point
        elif 'DHDSTformer_torso2' in args.model:
            gt = gt[:, [0, 1, 4, 8, 11, 14], :]
        elif 'DHDSTformer_right_arm' == args.model:
            gt = gt[:, [14, 15, 16], :]
        elif ('DHDSTformer_right_arm2' == args.model) or ('DHDSTformer_right_arm3' == args.model):
            gt = gt[:, [0, 14, 15, 16], :]
        
        if not args.mpjpe_after_part:
            err1_per_joint = mpjpe_for_each_joint(pred, gt) # (243, 17)
            err2_per_joint = p_mpjpe_for_each_joint(pred, gt) # (243, 17)
        
        for part in part_list:
            if part == 'whole': joint_list = [j for j in range(pred.shape[1])]
            elif part == 'torso_small': joint_list = [pelvis, r_hip, l_hip, neck, l_shoulder, r_shoulder]
            elif part == 'torso_full': joint_list = [pelvis, r_hip, l_hip, torso, neck, nose, head, l_shoulder, r_shoulder]
            elif part == 'torso_full_to_small': joint_list = [0, 1, 2, 4, 7, 8]
            elif part == 'arms': joint_list = [l_elbow, l_wrist, r_elbow, r_wrist]
            elif part == 'right_arm': joint_list = [r_shoulder, r_elbow, r_wrist]
            elif part == 'legs': joint_list = [r_knee, r_ankle, l_knee, l_ankle]
            elif part == 'pelvis': joint_list = [pelvis]
            elif part == 'r_hip': joint_list = [r_hip]
            elif part == 'l_hip': joint_list = [l_hip]
            elif part == 'torso': joint_list = [torso]
            elif part == 'neck': joint_list = [neck]
            elif part == 'l_shoulder': joint_list = [l_shoulder]
            elif part == 'r_shoulder': joint_list = [r_shoulder]
            elif part == 'l_elbow': joint_list = [l_elbow]
            elif part == 'l_wrist': joint_list = [l_wrist]
            elif part == 'r_elbow': joint_list = [r_elbow]
            elif part == 'r_wrist': joint_list = [r_wrist]
            elif part == 'r_knee' : joint_list = [r_knee]
            elif part == 'r_ankle': joint_list = [r_ankle]
            elif part == 'l_knee' : joint_list = [l_knee]
            elif part == 'l_ankle': joint_list = [l_ankle]
            elif part == 'nose'   : joint_list = [nose]
            elif part == 'head'   : joint_list = [head]
            
            if args.mpjpe_after_part:
                err1_per_joint = mpjpe_for_each_joint(pred[:, joint_list], gt[:, joint_list]) # (243, 17)
                err2_per_joint = p_mpjpe_for_each_joint(pred[:, joint_list], gt[:, joint_list]) # (243, 17)
                err1 = np.mean(err1_per_joint, axis=1)
                err2 = np.mean(err2_per_joint, axis=1)
            else:
                err1 = np.mean(err1_per_joint[:, joint_list], axis=1) # mpjpe(pred, gt) # (243, )
                err2 = np.mean(err2_per_joint[:, joint_list], axis=1) # p_mpjpe(pred, gt)
            total_result_dict[part]['e1_all'][frame_list] += err1 # (243, ) # 각 프레임 별 에러를 더해줌
            total_result_dict[part]['e2_all'][frame_list] += err2 # (243, ) # 각 프레임 별 에러를 더해줌
            total_result_dict[part]['oc'][frame_list] += 1
    
    # Error per action
    for idx in range(num_test_frames):
        for part in part_list:
            if total_result_dict[part]['e1_all'][idx] > 0:
                err1 = total_result_dict[part]['e1_all'][idx] / total_result_dict[part]['oc'][idx]
                err2 = total_result_dict[part]['e2_all'][idx] / total_result_dict[part]['oc'][idx]
                action = actions[idx]
                total_result_dict[part]['results'][action].append(err1)
                total_result_dict[part]['results_procrustes'][action].append(err2)

    for part in part_list:
        print('Part:', part)
        final_result = []
        final_result_procrustes = []
        summary_table = prettytable.PrettyTable()
        summary_table.field_names = ['test_name'] + action_names # first row
        for action in action_names:
            final_result.append(np.mean(total_result_dict[part]['results'][action]))
            final_result_procrustes.append(np.mean(total_result_dict[part]['results_procrustes'][action]))
        summary_table.add_row(['P1 ({})'.format(part)] + final_result) # second row
        summary_table.add_row(['P2 ({})'.format(part)] + final_result_procrustes) # third row
        print(summary_table)
        
        # Total Error
        e1_part = np.mean(np.array(final_result))
        e2_part = np.mean(np.array(final_result_procrustes))
        print('Protocol #1 Error (MPJPE):', e1_part, 'mm')
        print('Protocol #2 Error (P-MPJPE):', e2_part, 'mm')
        print('----------------------------------------')
        if part == args.eval_part:
            e1 = e1_part
            e2 = e2_part
    return e1, e2, results_all, total_result_dict
        
def train_epoch(args, model_pos, train_loader, losses, optimizer, has_3d, has_gt):
    model_pos.train()
    pbar = tqdm(train_loader)
    for (batch_input, batch_gt) in pbar:    
        # Preprocessing
        batch_size = len(batch_input)        
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
        if batch_gt.shape[2] == 17:
            batch_gt_torso = batch_gt[:, :, [0, 1, 4, 7, 8, 9, 10, 11, 14], :]
            batch_gt_limb = batch_gt[:, :, [2, 3, 5, 6, 12, 13, 15, 16], :]
        with torch.no_grad():
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if not has_3d:
                conf = copy.deepcopy(batch_input[:,:,:,2:])    # For 2D data, weight/confidence is at the last channel
            if args.rootrel:
                batch_gt = batch_gt - batch_gt[:,:,0:1,:] # move the pelvis to the origin
            else:
                batch_gt[:,:,:,2] = batch_gt[:,:,:,2] - batch_gt[:,0:1,0:1,2] # Place the depth of first frame root to 0. -> 첫번째 프레임의 depth를 0으로 설정
            if args.mask or args.noise:
                batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)
            if args.canonical:
                batch_input = batch_input - batch_input[:, :, 0:1, :] # root-relative
                batch_gt = batch_gt - batch_gt[:, :, 0:1, :]
        # Predict 3D poses
        if 'DHDSTformer_total' in args.model:
            predicted_3d_pos = model_pos(batch_input, length_type=args.train_length_type, ref_frame=args.length_frame)
        elif 'DHDSTformer_limb' in args.model:
            predicted_3d_pos = model_pos(batch_input, batch_gt_torso)
            pred_limb_pos = predicted_3d_pos[:, :, [2, 3, 5, 6, 12, 13, 15, 16], :]
        elif 'DHDSTformer_torso' in args.model:
            # get labels
            batch_lower_origin, batch_lower_R = get_batch_lower_torso_frame_from_pose(batch_gt)
            batch_upper_origin, batch_upper_R = get_batch_upper_torso_frame_from_pose(batch_gt)
            batch_lower_quat = matrix_to_quaternion(batch_lower_R)
            batch_upper_quat = matrix_to_quaternion(batch_upper_R)
            pred_torso, pred_lower_frame_R, pred_upper_frame_R = model_pos(batch_input)
            pred_lower_quat = matrix_to_quaternion(pred_lower_frame_R)
            pred_upper_quat = matrix_to_quaternion(pred_upper_frame_R)
        elif 'DHDSTformer_torso2' in args.model:
            pred_torso = model_pos(batch_input)
            batch_gt_torso = batch_gt[:, :, [0, 1, 4, 8, 11, 14], :] # r_hip, l_hip, neck, l_shoulder, r_shoulder
        elif 'DHDST_onevec' in args.model:
            input, gt_root_point, gt_length, gt_angle = get_input_gt_for_onevec(batch_input, batch_gt)
            pred_root_point, pred_length, pred_angle = model_pos(input)
            gt_R_azim = gt_angle[:, :, 0] # torch.Size([B, F])
            gt_R_elev = gt_angle[:, :, 1] # torch.Size([B, F])
            pred_R_azim = pred_angle[:, :, 0] # torch.Size([B, F])
            pred_R_elev = pred_angle[:, :, 1] # torch.Size([B, F])
            gt_child_point = batch_azim_elev_to_vec(gt_R_azim, gt_R_elev, gt_length, gt_root_point) # torch.Size([B, F, 3])
            pred_child_point = batch_azim_elev_to_vec(pred_R_azim, pred_R_elev, pred_length, pred_root_point) # torch.Size([B, F, 3])
            gt_3d_pos = torch.cat([gt_root_point.unsqueeze(2), gt_child_point.unsqueeze(2)], dim=2) # torch.Size([B, F, 2, 3])
            pred_3d_pos = torch.cat([pred_root_point.unsqueeze(2), pred_child_point.unsqueeze(2)], dim=2) # torch.Size([B, F, 2, 3])
            if args.rootrel:
                gt_3d_pos = gt_root_point - gt_3d_pos[:, :, 0:1, :]
            else:
                gt_3d_pos[:, :, :, 2] = gt_3d_pos[:, :, :, 2] - gt_3d_pos[:, 0:1, 0:1, 2] 
            
        elif 'DHDSTformer_right_arm' == args.model:
            batch_gt_limb = batch_gt[:, :, [14, 15, 16], :]
            pred_limb_pos = model_pos(batch_input)
        elif ('DHDSTformer_right_arm2' == args.model) or ('DHDSTformer_right_arm3' == args.model):
            batch_gt_limb = batch_gt[:, :, [0, 14, 15, 16], :]
            pred_limb_pos = model_pos(batch_input) 
        else:
            predicted_3d_pos = model_pos(batch_input)    # (N, T, 17, 3)
        optimizer.zero_grad()
        if has_3d:
            loss_total = 0
            if args.lambda_3d_pos > 0:
                loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
                loss_total += args.lambda_3d_pos * loss_3d_pos
                losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
            if args.lambda_scale > 0:
                loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
                loss_total += args.lambda_scale * loss_3d_scale
                losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
            if args.lambda_3d_velocity > 0:
                loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
                loss_total += args.lambda_3d_velocity * loss_3d_velocity
                losses['3d_vel'].update(loss_3d_velocity.item(), batch_size)
            if args.lambda_limb_pos > 0: 
                loss_3d_pos_limb = loss_mpjpe(pred_limb_pos, batch_gt_limb)
                loss_total += args.lambda_limb_pos * loss_3d_pos_limb
                losses['3d_pos_limb'].update(loss_3d_pos_limb.item(), batch_size)
            if args.lambda_limb_scale > 0:
                loss_3d_scale_limb = n_mpjpe(pred_limb_pos, batch_gt_limb)
                loss_total += args.lambda_limb_scale * loss_3d_scale_limb
                losses['3d_scale_limb'].update(loss_3d_scale_limb.item(), batch_size)
            if args.lambda_limb_velocity > 0:
                loss_3d_velocity_limb = loss_velocity(pred_limb_pos, batch_gt_limb)
                loss_total += args.lambda_limb_velocity * loss_3d_velocity_limb
                losses['3d_vel_limb'].update(loss_3d_velocity_limb.item(), batch_size)
            if args.lambda_torso_pos > 0:
                loss_3d_pos_torso = loss_mpjpe(pred_torso, batch_gt_torso)
                loss_total += args.lambda_torso_pos * loss_3d_pos_torso
                losses['3d_pos_torso'].update(loss_3d_pos_torso.item(), batch_size)
            if args.lambda_lower_frame_R > 0:
                loss_lower_frame_R = loss_mpjpe(pred_lower_quat, batch_lower_quat)
                loss_total += args.lambda_lower_frame_R * loss_lower_frame_R
                losses['lower_frame_R'].update(loss_lower_frame_R.item(), batch_size)
            if args.lambda_upper_frame_R > 0:
                loss_upper_frame_R = loss_mpjpe(pred_upper_quat, batch_upper_quat)
                loss_total += args.lambda_upper_frame_R * loss_upper_frame_R
                losses['upper_frame_R'].update(loss_upper_frame_R.item(), batch_size)
            if args.lambda_lv > 0:
                loss_lv = loss_limb_var(predicted_3d_pos)
                loss_total += args.lambda_lv * loss_lv
                losses['lv'].update(loss_lv.item(), batch_size)
            if args.lambda_lg > 0:
                loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
                loss_total += args.lambda_lg * loss_lg
                losses['lg'].update(loss_lg.item(), batch_size)
            if args.lambda_a > 0:
                loss_a = loss_angle(predicted_3d_pos, batch_gt)
                loss_total += args.lambda_a * loss_a
                losses['angle'].update(loss_a.item(), batch_size)
            if args.lambda_av > 0:
                loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
                loss_total += args.lambda_av * loss_av
                losses['angle_vel'].update(loss_av.item(), batch_size)
            if args.lambda_sym > 0:
                loss_sym = loss_symmetry(predicted_3d_pos)
                loss_total += args.lambda_sym * loss_sym
                losses['sym'].update(loss_sym.item(), batch_size)
            if args.lambda_root_point > 0:
                assert pred_root_point.shape == gt_root_point.shape, 'Root point shape mismatch'
                loss_root_point = loss_mpjpe(pred_root_point, gt_root_point)
                loss_total += args.lambda_root_point * loss_root_point
                losses['root_point'].update(loss_root_point.item(), batch_size)
            if args.lambda_length > 0:
                assert pred_length.shape == gt_length.shape, 'Length shape mismatch'
                loss_length = torch.mean(torch.norm(pred_length - gt_length, dim=len(pred_length.shape)-1))
                loss_total += args.lambda_length * loss_length
                losses['length'].update(loss_length.item(), batch_size)
            if args.lambda_angle > 0:
                assert pred_angle.shape == gt_angle.shape, 'Angle shape mismatch'
                loss_angle = torch.mean(torch.norm(pred_angle - gt_angle, dim=len(pred_angle.shape)-1))
                loss_total += args.lambda_angle * loss_angle
                losses['angle'].update(loss_angle.item(), batch_size)
            if args.lambda_onevec_pos:
                loss_onevec_pos = loss_mpjpe(pred_3d_pos, gt_3d_pos)
                loss_total += args.lambda_onevec_pos * loss_onevec_pos
                losses['onevec_pos'].update(loss_onevec_pos.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        else:
            loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
            loss_total = loss_2d_proj
            losses['2d_proj'].update(loss_2d_proj.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        loss_total.backward() # backprop
        optimizer.step()
        
        pbar.set_postfix({key: value.avg for key, value in losses.items()})
        try:
            if args.test_run:
                break
        except:
            pass

def train_with_config(args, opts):
    if not opts.evaluate:
        try:
            os.makedirs(opts.checkpoint)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
        train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))

    # Load dataset
    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 12,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 12,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    train_loader_3d = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)
    
    if args.train_2d:
        posetrack = PoseTrackDataset2D()
        posetrack_loader_2d = DataLoader(posetrack, **trainloader_params)
        instav = InstaVDataset2D()
        instav_loader_2d = DataLoader(instav, **trainloader_params)
    for subset in args.subset_list:
        if 'H36M' in subset:  datareader = DataReaderH36M(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = 'data/motion3d', dt_file=args.dt_file)
        elif 'AIHUB' in subset: datareader = DataReaderAIHUB(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = 'data/motion3d', dt_file=args.dt_file)
        elif 'FIT3D' in subset: datareader = DataReaderFIT3D(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = 'data/motion3d', dt_file=args.dt_file)
        elif 'KOOKMIN' in subset: datareader = DataReaderKOOKMIN(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = 'data/motion3d', dt_file=args.dt_file)
    min_loss = 100000
    
    # Load model
    if opts.pretrained:
        print('Loading checkpoint', opts.pretrained)
        chk_filename = os.path.join(opts.pretrained, opts.selection)
    else:
        chk_filename = ''

    print(args.model)
    if 'DHDSTformer_total' in args.model: model_pos = DHDSTformer_total(chk_filename=chk_filename, args=args)
    elif 'DHDSTformer_total2' in args.model: model_pos = DHDSTformer_total2(chk_filename=chk_filename, args=args)
    elif 'DHDSTformer_total3' in args.model: model_pos = DHDSTformer_total3(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDSTformer_torso': model_pos = DHDSTformer_torso(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDSTformer_torso2': model_pos = DHDSTformer_torso2(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDSTformer_limb': model_pos = DHDSTformer_limb(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDSTformer_limb2': model_pos = DHDSTformer_limb2(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDSTformer_limb3': model_pos = DHDSTformer_limb3(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDSTformer_limb4': model_pos = DHDSTformer_limb4(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDSTformer_limb5': model_pos = DHDSTformer_limb5(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDSTformer_torso_limb': model_pos = DHDSTformer_torso_limb(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDST_onevec': model_pos = DHDSTformer_onevec(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDSTformer_right_arm': model_pos = DHDSTformer_right_arm(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDSTformer_right_arm2': model_pos = DHDSTformer_right_arm2(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDSTformer_right_arm3': model_pos = DHDSTformer_right_arm3(chk_filename=chk_filename, args=args)
    else: 
        model_pos = load_backbone(args)
        if args.finetune:
            if opts.resume:
                chk_filename = opts.resume
                print('Loading checkpoint', chk_filename)
                checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
                model_pos.load_state_dict(checkpoint['model_pos'], strict=False)
            elif opts.pretrained:
                chk_filename = os.path.join(opts.pretrained, opts.selection)
                print('Loading checkpoint', chk_filename)
                checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
                model_pos.load_state_dict(checkpoint['model_pos'], strict=False)    
        else:
            chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
            if os.path.exists(chk_filename):
                opts.resume = chk_filename
            if opts.resume:
                chk_filename = opts.resume
                print('Loading checkpoint', chk_filename)
                checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
                model_pos.load_state_dict(checkpoint['model_pos'], strict=False)
    
    if torch.cuda.is_available():
        model_pos = nn.DataParallel(model_pos)
        model_pos = model_pos.cuda()
    
    if opts.evaluate:
        chk_filename = os.path.join(opts.checkpoint, opts.evaluate)
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model_pos.load_state_dict(checkpoint['model_pos'], strict=False)
        
    if args.finetune_only_head:
        for name, parameter in model_pos.named_parameters():
            if ('head' in name) or ('pre_logits' in name): 
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

    model_params = 0
    for parameter in model_pos.parameters():
        if parameter.requires_grad:
            model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
        
    if args.partial_train:
        model_pos = partial_train_layers(model_pos, args.partial_train)

    if not opts.evaluate: # training process 
        lr = args.learning_rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr, weight_decay=args.weight_decay)
        lr_decay = args.lr_decay
        st = 0
        if args.train_2d:
            print('INFO: Training on {}(3D)+{}(2D) batches'.format(len(train_loader_3d), len(instav_loader_2d) + len(posetrack_loader_2d)))
        else:
            print('INFO: Training on {}(3D) batches'.format(len(train_loader_3d)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')            
            lr = checkpoint['lr']
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                min_loss = checkpoint['min_loss']
                
        args.mask = (args.mask_ratio > 0 and args.mask_T_ratio > 0)
        if args.mask or args.noise:
            args.aug = Augmenter2D(args)
        
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            start_time = time()
            losses = {}
            if args.lambda_3d_pos > 0:        losses['3d_pos'] = AverageMeter()
            if args.lambda_scale > 0:         losses['3d_scale'] = AverageMeter()
            if args.lambda_3d_velocity > 0:   losses['3d_vel'] = AverageMeter()
            if args.lambda_limb_pos > 0:      losses['3d_pos_limb'] = AverageMeter()
            if args.lambda_limb_scale > 0:    losses['3d_scale_limb'] = AverageMeter()
            if args.lambda_limb_velocity > 0: losses['3d_vel_limb'] = AverageMeter()
            if args.lambda_torso_pos > 0:     losses['3d_pos_torso'] = AverageMeter()
            if args.lambda_lower_frame_R > 0: losses['lower_frame_R'] = AverageMeter()
            if args.lambda_upper_frame_R > 0: losses['upper_frame_R'] = AverageMeter()
            if args.lambda_lg > 0:            losses['lg'] = AverageMeter()
            if args.lambda_lv > 0:            losses['lv'] = AverageMeter()
            if args.lambda_a > 0:             losses['angle'] = AverageMeter()
            if args.lambda_av > 0:            losses['angle_vel'] = AverageMeter()
            if args.lambda_sym > 0:           losses['sym'] = AverageMeter()
            if args.lambda_root_point > 0:    losses['root_point'] = AverageMeter()
            if args.lambda_length > 0:        losses['length'] = AverageMeter()
            if args.lambda_angle > 0:         losses['angle'] = AverageMeter()
            if args.lambda_onevec_pos:        losses['onevec_pos'] = AverageMeter()
            losses['total'] = AverageMeter()
            losses['2d_proj'] = AverageMeter()
            #N = 0
                        
            # Curriculum Learning
            if args.train_2d and (epoch >= args.pretrain_3d_curriculum):
                train_epoch(args, model_pos, posetrack_loader_2d, losses, optimizer, has_3d=False, has_gt=True)
                train_epoch(args, model_pos, instav_loader_2d, losses, optimizer, has_3d=False, has_gt=False)
            train_epoch(args, model_pos, train_loader_3d, losses, optimizer, has_3d=True, has_gt=True) 
            elapsed = (time() - start_time) / 60

            if '3d_pos' in losses:
                loss_print = losses['3d_pos'].avg
            else:
                loss_print = losses[list(losses.keys())[0]].avg
            if args.no_eval:
                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    loss_print))
            else:
                if 'DHDST_onevec' in args.model:
                    e1, e2, results_all, total_result_dict = evaluate_onevec(args, model_pos, test_loader, datareader)
                else:
                    e1, e2, results_all, total_result_dict = evaluate(args, model_pos, test_loader, datareader)
                print('[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    loss_print,
                    e1, e2))
                train_writer.add_scalar('Error P1', e1, epoch + 1)
                train_writer.add_scalar('Error P2', e2, epoch + 1)
                train_writer.add_scalar('lr', lr, epoch + 1)
                train_writer.add_scalar('loss_total', losses['total'].avg, epoch + 1)
                train_writer.add_scalar('loss_2d_proj', losses['2d_proj'].avg, epoch + 1)
                if args.lambda_3d_pos > 0:        train_writer.add_scalar('loss_3d_pos', losses['3d_pos'].avg, epoch + 1)
                if args.lambda_scale > 0:         train_writer.add_scalar('loss_3d_scale', losses['3d_scale'].avg, epoch + 1)
                if args.lambda_3d_velocity > 0:   train_writer.add_scalar('loss_3d_velocity', losses['3d_vel'].avg, epoch + 1)
                if args.lambda_limb_pos > 0:      train_writer.add_scalar('loss_3d_pos_limb', losses['3d_pos_limb'].avg, epoch + 1)
                if args.lambda_limb_scale > 0:    train_writer.add_scalar('loss_3d_scale_limb', losses['3d_scale_limb'].avg, epoch + 1)
                if args.lambda_limb_velocity > 0: train_writer.add_scalar('loss_3d_velocity_limb', losses['3d_vel_limb'].avg, epoch + 1)
                if args.lambda_torso_pos > 0:     train_writer.add_scalar('loss_3d_pos_torso', losses['3d_pos_torso'].avg, epoch + 1)
                if args.lambda_lower_frame_R > 0: train_writer.add_scalar('loss_lower_frame_R', losses['lower_frame_R'].avg, epoch + 1)
                if args.lambda_upper_frame_R > 0: train_writer.add_scalar('loss_upper_frame_R', losses['upper_frame_R'].avg, epoch + 1)
                if args.lambda_lv > 0:            train_writer.add_scalar('loss_lv', losses['lv'].avg, epoch + 1)
                if args.lambda_lg > 0:            train_writer.add_scalar('loss_lg', losses['lg'].avg, epoch + 1)
                if args.lambda_a > 0:             train_writer.add_scalar('loss_a', losses['angle'].avg, epoch + 1)
                if args.lambda_av > 0:            train_writer.add_scalar('loss_av', losses['angle_vel'].avg, epoch + 1)
                if args.lambda_sym > 0:           train_writer.add_scalar('loss_sym', losses['sym'].avg, epoch + 1)
                if args.lambda_root_point > 0:    train_writer.add_scalar('loss_root_point', losses['root_point'].avg, epoch + 1)
                if args.lambda_length > 0:        train_writer.add_scalar('loss_length', losses['length'].avg, epoch + 1)
                if args.lambda_angle > 0:         train_writer.add_scalar('loss_angle', losses['angle'].avg, epoch + 1)
                if args.lambda_onevec_pos:        train_writer.add_scalar('loss_onevec_pos', losses['onevec_pos'].avg, epoch + 1)
                if 'arms' in args.part_list:
                    arm_mpjpe = np.mean([np.mean(total_result_dict['arms']['results'][key]) for key in total_result_dict['arms']['results'].keys()])
                    arm_mpjpe_procrustes = np.mean([np.mean(total_result_dict['arms']['results_procrustes'][key]) for key in total_result_dict['arms']['results_procrustes'].keys()])
                    train_writer.add_scalar('arm P1', arm_mpjpe, epoch + 1)
                    train_writer.add_scalar('arm P2', arm_mpjpe_procrustes, epoch + 1)
                if 'legs' in args.part_list:
                    leg_mpjpe = np.mean([np.mean(total_result_dict['legs']['results'][key]) for key in total_result_dict['legs']['results'].keys()])
                    leg_mpjpe_procrustes = np.mean([np.mean(total_result_dict['legs']['results_procrustes'][key]) for key in total_result_dict['legs']['results_procrustes'].keys()])
                    train_writer.add_scalar('leg P1', leg_mpjpe, epoch + 1)
                    train_writer.add_scalar('leg P2', leg_mpjpe_procrustes, epoch + 1)
                
            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            # Save checkpoints
            chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
            chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin')
            
            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, min_loss)
            if (epoch + 1) % args.checkpoint_frequency == 0:
                save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss)
            if e1 < min_loss:
                min_loss = e1
                save_checkpoint(chk_path_best, epoch, lr, optimizer, model_pos, min_loss)
            try:
                if args.test_run:
                    break
            except:
                pass

    if opts.evaluate:
        if 'DHDST_onevec' in args.model:
            evaluate_onevec(args, model_pos, test_loader, datareader)
        else:
            e1, e2, results_all, total_result_dict = evaluate(args, model_pos, test_loader, datareader)

def check_args(args):
    try: test = args.model
    except:
        if opts.evaluate: args.model = opts.evaluate
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
    try: test = args.lambda_angle
    except: args.lambda_angle = 0.0
    try: test = args.lambda_onevec_pos
    except: args.lambda_onevec_pos = 0.0

    try: test = args.canonical
    except: args.canonical = False
    
    try: test = args.finetune_only_head
    except: args.finetune_only_head = False

    try: test = args.mpjpe_after_part
    except: args.mpjpe_after_part = False

    return args

if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    if type(opts.part_list) != type(None):
        args.part_list = opts.part_list
    # check arguments
    args = check_args(args)
    print(args)
    train_with_config(args, opts)