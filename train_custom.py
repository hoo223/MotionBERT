import re
import sys
sys.path.append('/home/hrai/codes/hpe_library')
from lib_import import *
from my_utils import *
os.chdir('/home/hrai/codes/MotionBERT')

import os
import numpy as np

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
from lib.utils.args import parse_args, check_args
from lib.utils.utils_data import flip_data
from lib.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.data.augmentation import Augmenter2D
from lib.data.datareader_h36m import DataReaderH36M
from lib.data.datareader_aihub import DataReaderAIHUB
from lib.data.datareader_fit3d import DataReaderFIT3D
from lib.data.datareader_kookmin import DataReaderKOOKMIN
from lib.model.loss import *
from lib.model.training import *
from lib.model.evaluation import *
from lib.model.DHDSTformer import DHDSTformer_total, DHDSTformer_total2, DHDSTformer_total3, DHDSTformer_total4, \
    DHDSTformer_limb, DHDSTformer_limb2, DHDSTformer_limb3, DHDSTformer_limb4, DHDSTformer_limb5, \
    DHDSTformer_right_arm, DHDSTformer_right_arm2, DHDSTformer_right_arm3, \
    DHDSTformer_torso, DHDSTformer_torso2, \
    DHDSTformer_torso_limb, \
    DHDSTformer_onevec

os.environ["NCCL_P2P_DISABLE"]= '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

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
    model_pos.eval()  
    # get inference results          
    results_all = inference_eval(args, model_pos, test_loader, datareader)
    # calculate evaluation metric
    e1, e2, total_result_dict = calculate_eval_metric(args, results_all, datareader)
    return e1, e2, results_all, total_result_dict
            
        
def train_epoch(args, model_pos, train_loader, losses, optimizer, has_3d, has_gt):
    model_pos.train()
    pbar = tqdm(train_loader)
    for (batch_input, batch_gt) in pbar:    
        batch_size = len(batch_input)        
        # preprocessing
        batch_input, batch_gt, batch_gt_torso, batch_gt_limb, conf = preprocess_train(args, batch_input, batch_gt, has_3d, has_gt)
        # inferece 3D poses
        if args.model in ['DHDSTformer_total', 'DHDSTformer_total2', 'DHDSTformer_total3']: 
            predicted_3d_pos, pred_angle, gt_angle = inference_train(args, model_pos, batch_input, batch_gt, batch_gt_torso)
        elif args.model in ['DHDSTformer_total4']:
            pred_torso, pred_dh_angle, pred_dh_length, pred_lower_frame_R, pred_upper_frame_R, predicted_3d_pos = inference_train(args, model_pos, batch_input, batch_gt, batch_gt_torso)
            pred_dh_length = pred_dh_length.reshape(-1, 8)
            pred_lower_quat = matrix_to_quaternion(pred_lower_frame_R)
            pred_upper_quat = matrix_to_quaternion(pred_upper_frame_R)
            # get frame label
            batch_lower_origin, batch_lower_R = get_batch_lower_torso_frame_from_pose(batch_gt)
            batch_upper_origin, batch_upper_R = get_batch_upper_torso_frame_from_pose(batch_gt)
            batch_lower_quat = matrix_to_quaternion(batch_lower_R)
            batch_upper_quat = matrix_to_quaternion(batch_upper_R)
            # get limb, angle label
            gt_dh_model = BatchDHModel(batch_gt, batch_size=batch_size, num_frames=args.clip_len, head=True)
            gt_dh_angle = gt_dh_model.get_batch_appendage_angles() # (B, F, 16)
            gt_dh_length = gt_dh_model.get_batch_appendage_length()[:, 0, :] # (B, 8)

        elif 'DHDSTformer_limb' in args.model:
            predicted_3d_pos, pred_limb_pos = inference_train(args, model_pos, batch_input, batch_gt, batch_gt_torso)
        elif 'DHDSTformer_torso' in args.model:
            pred_torso, batch_gt_torso, pred_lower_frame_R, pred_upper_frame_R = inference_train(args, model_pos, batch_input, batch_gt, batch_gt_torso)
            pred_lower_quat = matrix_to_quaternion(pred_lower_frame_R)
            pred_upper_quat = matrix_to_quaternion(pred_upper_frame_R)
            # get frame label
            batch_lower_origin, batch_lower_R = get_batch_lower_torso_frame_from_pose(batch_gt)
            batch_upper_origin, batch_upper_R = get_batch_upper_torso_frame_from_pose(batch_gt)
            batch_lower_quat = matrix_to_quaternion(batch_lower_R)
            batch_upper_quat = matrix_to_quaternion(batch_upper_R)
            
        elif 'DHDSTformer_torso2' in args.model:
            pred_torso, batch_gt_torso = inference_train(args, model_pos, batch_input, batch_gt, batch_gt_torso)
        elif 'DHDST_onevec' in args.model:
            pred_3d_pos, gt_3d_pos, pred_root_point, gt_root_point, pred_length, gt_length = \
                inference_train(args, model_pos, batch_input, batch_gt, batch_gt_torso)
        elif 'DHDSTformer_right_arm' == args.model:
            pred_limb_pos, batch_gt_limb = inference_train(args, model_pos, batch_input, batch_gt, batch_gt_torso)
        elif ('DHDSTformer_right_arm2' == args.model) or ('DHDSTformer_right_arm3' == args.model):
            pred_limb_pos, batch_gt_limb = inference_train(args, model_pos, batch_input, batch_gt, batch_gt_torso)
        else:
            predicted_3d_pos, pred_angle, gt_angle = inference_train(args, model_pos, batch_input, batch_gt, batch_gt_torso)
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
            if args.lambda_dh_angle > 0:
                assert pred_angle.shape == gt_angle.shape, 'Angle shape mismatch'
                loss_dh_angle = nn.L1Loss()(pred_angle, gt_angle) # torch.mean(torch.norm(pred_angle - gt_angle, dim=len(pred_angle.shape)-1))
                loss_total += args.lambda_dh_angle * loss_dh_angle
                losses['dh_angle'].update(loss_dh_angle.item(), batch_size)
            if args.lambda_onevec_pos:
                loss_onevec_pos = loss_mpjpe(pred_3d_pos, gt_3d_pos)
                loss_total += args.lambda_onevec_pos * loss_onevec_pos
                losses['onevec_pos'].update(loss_onevec_pos.item(), batch_size)
            if args.lambda_dh_angle2 > 0:
                loss_dh_angle2 = nn.MSELoss()(pred_dh_angle, gt_dh_angle)
                loss_total += args.lambda_dh_angle2 * loss_dh_angle2
                losses['dh_angle2'].update(loss_dh_angle2.item(), batch_size)
            if args.lambda_dh_length > 0:
                loss_dh_length = nn.MSELoss()(pred_dh_length, gt_dh_length)
                loss_total += args.lambda_dh_length * loss_dh_length
                losses['dh_length'].update(loss_dh_length.item(), batch_size)
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
        if 'H36M' in subset:  datareader = DataReaderH36M(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = 'data/motion3d', dt_file=args.dt_file, mode=args.gt_mode)
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
    if 'DHDSTformer_total' == args.model: model_pos = DHDSTformer_total(chk_filename=chk_filename, args=args)
    elif 'DHDSTformer_total2' == args.model: model_pos = DHDSTformer_total2(chk_filename=chk_filename, args=args)
    elif 'DHDSTformer_total3' == args.model: model_pos = DHDSTformer_total3(chk_filename=chk_filename, args=args)
    elif 'DHDSTformer_total4' == args.model: model_pos = DHDSTformer_total4(args=args)
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
            if args.lambda_dh_angle > 0:      losses['dh_angle'] = AverageMeter()
            if args.lambda_onevec_pos:        losses['onevec_pos'] = AverageMeter()
            if args.lambda_dh_angle2 > 0:     losses['dh_angle2'] = AverageMeter()
            if args.lambda_dh_length > 0:     losses['dh_length'] = AverageMeter()
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
                if args.lambda_dh_angle > 0:      train_writer.add_scalar('loss_dh_angle', losses['dh_angle'].avg, epoch + 1)
                if args.lambda_onevec_pos:        train_writer.add_scalar('loss_onevec_pos', losses['onevec_pos'].avg, epoch + 1)
                if args.lambda_dh_angle2 > 0:     train_writer.add_scalar('loss_dh_angle2', losses['dh_angle2'].avg, epoch + 1)
                if args.lambda_dh_length > 0:     train_writer.add_scalar('loss_dh_length', losses['dh_length'].avg, epoch + 1)
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

if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    if type(opts.part_list) != type(None):
        args.part_list = opts.part_list
    # check arguments
    args = check_args(args, opts)
    print(args)
    train_with_config(args, opts)