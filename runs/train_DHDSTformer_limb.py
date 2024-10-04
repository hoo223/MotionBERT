import os
import numpy as np
import argparse
import errno
import math
import pickle
import tensorboardX
from tqdm import tqdm

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
from lib.data.datareader_kookmin import DataReaderKOOKMIN
from lib.model.loss import *
from lib.model.DHDSTformer import DHDSTformer_limb, DHDSTformer_limb2, DHDSTformer_limb3, DHDSTformer_limb4, DHDSTformer_limb5

sys.path.append('/home/hrai/codes/PoseAdaptor')
from hpe_library.lib_import import *
from hpe_library. my_utils import *
os.chdir('/home/hrai/codes/MotionBERT')
from time import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-g', '--gpu', default='0, 1', type=str, help='GPU id')
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
    
def evaluate(args, model_pos, test_loader, datareader):
    print('INFO: Testing')
    results_all = []
    model_pos.eval()            
    print(args.flip)
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader):
            # N, T = batch_gt.shape[:2] # batch_size, 243
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
                batch_gt = batch_gt.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            batch_gt_torso = batch_gt[:, :, [0, 1, 4, 7, 8, 9, 10, 11, 14], :] 
                
            # inference
            if args.flip:    
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input, batch_gt_torso)
                predicted_3d_pos_flip = model_pos(batch_input_flip, batch_gt_torso)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)                   # Flip back
                predicted_3d_pos = (predicted_3d_pos_1+predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model_pos(batch_input, batch_gt_torso)
                
            # if args.rootrel:
            #     predicted_3d_pos[:,:,0,:] = 0     # [N,T,17,3]
            # else:
            #     batch_gt[:,0,0,2] = 0
            # if args.gt_2d:
            #     predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())
            # try:
            #     if args.test_run:
            #         break
            # except:
            #     pass
    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)

    #np.save('/home/hrai/codes/MotionBERT/custom_codes/h36m_result_denormalized_.npy', results_all)

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
    
    total_result_dict = {}
    pelvis, r_hip, l_hip, torso, neck, l_shoulder, r_shoulder = 0, 1, 4, 7, 8, 11, 14
    r_knee, r_ankle, l_knee, l_ankle = 2, 3, 5, 6
    l_elbow, l_wrist, r_elbow, r_wrist = 12, 13, 15, 16
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    block_list = ['s_09_act_05_subact_02', 
                  's_09_act_10_subact_02', 
                  's_09_act_13_subact_01']
    part_list = ['arms', 'legs'] # ['whole', 'torso', 'arms', 'legs', 'pelvis', 'r_hip', 'l_hip', 'torso', 'neck', 'l_shoulder', 'r_shoulder', 'l_elbow', 'l_wrist', 'r_elbow', 'r_wrist', 'r_knee', 'r_ankle', 'l_knee', 'l_ankle']
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
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx] # [0 1 ... 242]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:,None,None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor # scaling image to world scale
        
        # Root-relative Errors
        pred = pred - pred[:,0:1,:] # (243, 17, 3)
        gt = gt - gt[:,0:1,:] # (243, 17, 3)
        
        err1_per_joint = mpjpe_for_each_joint(pred, gt) # (243, 17)
        err2_per_joint = p_mpjpe_for_each_joint(pred, gt) # (243, 17)
        
        for part in part_list:
            if part == 'whole': joint_list = [j for j in range(17)]
            elif part == 'torso': joint_list = [pelvis, r_hip, l_hip, torso, neck, l_shoulder, r_shoulder]
            elif part == 'arms': joint_list = [l_elbow, l_wrist, r_elbow, r_wrist]
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
        e1 = np.mean(np.array(final_result))
        e2 = np.mean(np.array(final_result_procrustes))
        print('Protocol #1 Error (MPJPE):', e1, 'mm')
        print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
        print('----------------------------------------')
    
    return e1, e2, results_all, total_result_dict
        
def train_epoch(args, model_pos, train_loader, losses, optimizer, has_3d, has_gt):
    model_pos.train()
    for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):    
        batch_size = len(batch_input)        
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
        batch_gt_torso = batch_gt[:, :, [0, 1, 4, 7, 8, 9, 10, 11, 14], :]
        batch_gt_limb_pos = batch_gt[:, :, [2, 3, 5, 6, 12, 13, 15, 16], :]
        with torch.no_grad():
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if not has_3d:
                conf = copy.deepcopy(batch_input[:,:,:,2:])    # For 2D data, weight/confidence is at the last channel
            if args.rootrel:
                batch_gt = batch_gt - batch_gt[:,:,0:1,:]
            else:
                batch_gt[:,:,:,2] = batch_gt[:,:,:,2] - batch_gt[:,0:1,0:1,2] # Place the depth of first frame root to 0.
            if args.mask or args.noise:
                batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)
        # Predict 3D poses
        predicted_3d_pos = model_pos(batch_input, batch_gt_torso)
        pred_limb_pos = predicted_3d_pos[:, :, [2, 3, 5, 6, 12, 13, 15, 16], :]
        #print('Time for forward pass:', end-start, '\n')

        optimizer.zero_grad()
        if has_3d:
            loss_3d_pos = loss_mpjpe(pred_limb_pos, batch_gt_limb_pos)
            loss_3d_scale = n_mpjpe(pred_limb_pos, batch_gt_limb_pos)
            loss_3d_velocity = loss_velocity(pred_limb_pos, batch_gt_limb_pos)
            # loss_lv = loss_limb_var(predicted_3d_pos)
            # limb_lens_x = get_limb_lens(predicted_3d_pos)[:, :, [1, 2, 4, 5, 11, 12, 14, 15]]
            # limb_lens_gt = get_limb_lens(batch_gt)[:, :, [1, 2, 4, 5, 11, 12, 14, 15]] 
            # loss_lg = nn.L1Loss()(limb_lens_x, limb_lens_gt)
            # loss_lg = loss_limb_gt(predicted_appendage_length, gt_appendage_length)
            # loss_a = loss_angle(predicted_appendage_angle, gt_appendage_angle)
            # loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
            # loss_sym = loss_symmetry(predicted_3d_pos)
            # loss_total = loss_3d_pos + \
            #              args.lambda_scale       * loss_3d_scale + \
            #              args.lambda_3d_velocity * loss_3d_velocity + \
            #              args.lambda_lv          * loss_lv + \
            #              args.lambda_lg          * loss_lg + \
            #              args.lambda_a           * loss_a  + \
            #              args.lambda_av          * loss_av + \
            #              args.lambda_sym         * loss_sym
            loss_total = args.lambda_pos         * loss_3d_pos + \
                         args.lambda_scale       * loss_3d_scale + \
                         args.lambda_3d_velocity * loss_3d_velocity 
                         # args.lambda_lg  * loss_lg
                         # args.lambda_a   * loss_a
                         
            losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
            losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
            losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
            # losses['lv'].update(loss_lv.item(), batch_size)
            # losses['lg'].update(loss_lg.item(), batch_size)
            # losses['angle'].update(loss_a.item(), batch_size)
            # losses['angle_velocity'].update(loss_av.item(), batch_size)
            # losses['sym'].update(loss_sym.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        else:
            loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
            loss_total = loss_2d_proj
            losses['2d_proj'].update(loss_2d_proj.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        loss_total.backward()
        optimizer.step()
        try:
            if args.test_run:
                break
        except:
            pass

def train_with_config(args, opts):
    torch.autograd.set_detect_anomaly(True)
    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))


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
        
    datareader = DataReaderKOOKMIN(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = 'data/motion3d', dt_file=args.dt_file)
    min_loss = 100000
    
    # model_backbone = load_backbone(args)
    # model_params = 0
    # for parameter in model_backbone.parameters():
    #     model_params = model_params + parameter.numel()
    # print('INFO: Trainable parameter count:', model_params)

    # if torch.cuda.is_available():
    #     model_backbone = nn.DataParallel(model_backbone)
    #     model_backbone = model_backbone.cuda()

    # if args.finetune:
    #     if opts.resume or opts.evaluate:
    #         chk_filename = opts.evaluate if opts.evaluate else opts.resume
    #         print('Loading checkpoint', chk_filename)
    #         checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    #         model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    #         model_pos = model_backbone
    #     else:
    #         chk_filename = os.path.join(opts.pretrained, opts.selection)
    #         print('Loading checkpoint', chk_filename)
    #         checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    #         model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    #         model_pos = model_backbone            
    # else:
    #     chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    #     if os.path.exists(chk_filename):
    #         opts.resume = chk_filename
    #     if opts.resume or opts.evaluate:
    #         chk_filename = opts.evaluate if opts.evaluate else opts.resume
    #         print('Loading checkpoint', chk_filename)
    #         checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    #         model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    #     model_pos = model_backbone
    
    if opts.pretrained:
        print('Loading checkpoint', opts.pretrained)
        chk_filename = os.path.join(opts.pretrained, opts.selection)
    else:
        chk_filename = ''
    if args.model == 'DHDSTformer_limb': model_pos = DHDSTformer_limb(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDSTformer_limb2': model_pos = DHDSTformer_limb2(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDSTformer_limb3': model_pos = DHDSTformer_limb3(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDSTformer_limb4': model_pos = DHDSTformer_limb4(chk_filename=chk_filename, args=args)
    elif args.model == 'DHDSTformer_limb5': model_pos = DHDSTformer_limb5(chk_filename=chk_filename, args=args)
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()
    
    if opts.evaluate:
        chk_filename = opts.evaluate
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model_pos.load_state_dict(checkpoint['model_pos'], strict=True)
        model_pos.eval()
        
    if args.partial_train:
        model_pos = partial_train_layers(model_pos, args.partial_train)

    if not opts.evaluate:        
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
            losses['3d_pos'] = AverageMeter()
            losses['3d_scale'] = AverageMeter()
            # losses['2d_proj'] = AverageMeter()
            # losses['lg'] = AverageMeter()
            # losses['lv'] = AverageMeter()
            losses['total'] = AverageMeter()
            losses['3d_velocity'] = AverageMeter()
            # losses['angle'] = AverageMeter()
            # losses['angle_velocity'] = AverageMeter()
            # losses['sym'] = AverageMeter()
            N = 0
                        
            # Curriculum Learning
            if args.train_2d and (epoch >= args.pretrain_3d_curriculum):
                train_epoch(args, model_pos, posetrack_loader_2d, losses, optimizer, has_3d=False, has_gt=True)
                train_epoch(args, model_pos, instav_loader_2d, losses, optimizer, has_3d=False, has_gt=False)
            train_epoch(args, model_pos, train_loader_3d, losses, optimizer, has_3d=True, has_gt=True) 
            elapsed = (time() - start_time) / 60

            if args.no_eval:
                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                   losses['3d_pos'].avg))
            else:
                e1, e2, results_all, total_result_dict = evaluate(args, model_pos, test_loader, datareader)
                print('[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses['3d_pos'].avg,
                    e1, e2))
                train_writer.add_scalar('Error P1', e1, epoch + 1)
                train_writer.add_scalar('Error P2', e2, epoch + 1)
                train_writer.add_scalar('lr', lr, epoch + 1)
                train_writer.add_scalar('loss_3d_pos', losses['3d_pos'].avg, epoch + 1)
                # train_writer.add_scalar('loss_2d_proj', losses['2d_proj'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_scale', losses['3d_scale'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_velocity', losses['3d_velocity'].avg, epoch + 1)
                # train_writer.add_scalar('loss_lv', losses['lv'].avg, epoch + 1)
                # train_writer.add_scalar('loss_lg', losses['lg'].avg, epoch + 1)
                # train_writer.add_scalar('loss_a', losses['angle'].avg, epoch + 1)
                # train_writer.add_scalar('loss_av', losses['angle_velocity'].avg, epoch + 1)
                # train_writer.add_scalar('loss_sym', losses['sym'].avg, epoch + 1)
                train_writer.add_scalar('loss_total', losses['total'].avg, epoch + 1)
                arm_mpjpe = np.mean([np.mean(total_result_dict['arms']['results'][key]) for key in total_result_dict['arms']['results'].keys()])
                arm_mpjpe_procrustes = np.mean([np.mean(total_result_dict['arms']['results_procrustes'][key]) for key in total_result_dict['arms']['results_procrustes'].keys()])
                leg_mpjpe = np.mean([np.mean(total_result_dict['legs']['results'][key]) for key in total_result_dict['legs']['results'].keys()])
                leg_mpjpe_procrustes = np.mean([np.mean(total_result_dict['legs']['results_procrustes'][key]) for key in total_result_dict['legs']['results_procrustes'].keys()])
                train_writer.add_scalar('arm P1', arm_mpjpe, epoch + 1)
                train_writer.add_scalar('arm P2', arm_mpjpe_procrustes, epoch + 1)
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
        e1, e2, results_all, total_result_dict = evaluate(args, model_pos, test_loader, datareader)

if __name__ == "__main__":
    opts = parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu # only one 4090 lacks memory
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    try:
        test = args.lambda_sym
    except:
        print('no lambda_sym')
        args.lambda_sym = 0
    train_with_config(args, opts)