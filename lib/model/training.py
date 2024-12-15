import os
import sys
import copy
import errno
import numpy as np
from time import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX
from lib.data.augmentation import Augmenter2D
from lib.utils.learning import * # partial_train_layers, AverageMeter

sys.path.append('/home/hrai/codes/hpe_library')
from hpe_library.my_utils import get_limb_angle, get_batch_lower_torso_frame_from_pose, get_batch_upper_torso_frame_from_pose, matrix_to_quaternion, batch_azim_elev_to_vec, get_input_gt_for_onevec, BatchDHModel
from lib.model.loss import *
from lib.model.evaluation import *
##
def save_checkpoint(chk_path, epoch, start_epoch, lr, optimizer, model_pos, min_loss):
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'start_epoch': start_epoch,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss' : min_loss
    }, chk_path)
##
def train(args, opts, checkpoint, model_pos, train_loader_3d, posetrack_loader_2d, instav_loader_2d, test_loader, datareader, run=None):
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))

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

    min_loss = 100000
    lr = args.learning_rate
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr, weight_decay=args.weight_decay)
    lr_decay = args.lr_decay
    st = 0
    args.start_epoch = 0
    if args.train_2d:
        print('INFO: Training on {}(3D)+{}(2D) batches'.format(len(train_loader_3d), len(instav_loader_2d) + len(posetrack_loader_2d)))
    else:
        print('INFO: Training on {}(3D) batches'.format(len(train_loader_3d)))
    if opts.resume:
        st = checkpoint['epoch']
        args.start_epoch = st
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
        # Initialize loss dict for storing loss values
        losses = generate_loss_dict(args)

        # Curriculum Learning
        if args.train_2d and (epoch >= args.pretrain_3d_curriculum):
            train_epoch(args, model_pos, posetrack_loader_2d, losses, optimizer, has_3d=False, has_gt=True)
            train_epoch(args, model_pos, instav_loader_2d, losses, optimizer, has_3d=False, has_gt=False)
        train_epoch(args, model_pos, train_loader_3d, losses, optimizer, has_3d=True, has_gt=True)
        elapsed = (time() - start_time) / 60

        if '3d_pos' in losses: loss_print = losses['3d_pos'].avg
        else: loss_print = losses[list(losses.keys())[0]].avg

        # Evaluation
        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (epoch + 1, elapsed, lr, loss_print))
        else:
            e1, e2, results_all, inputs_all, gts_all, total_result_dict = evaluate(args, model_pos, test_loader, datareader, checkpoint)
            print('[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (epoch + 1, elapsed, lr, loss_print, e1, e2))

        # Update tensorboard
        train_writer = update_train_writer(args, train_writer, losses, e1, e2, lr, epoch, total_result_dict, run)

        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

        # Save checkpoints
        chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
        chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
        chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin')
        save_checkpoint(chk_path_latest, epoch, args.start_epoch, lr, optimizer, model_pos, min_loss) # save latest checkpoint
        if (epoch + 1) % args.checkpoint_frequency == 0:
            save_checkpoint(chk_path, epoch, args.start_epoch, lr, optimizer, model_pos, min_loss) # save checkpoint every args.checkpoint_frequency epochs
        if e1 < min_loss:
            min_loss = e1
            save_checkpoint(chk_path_best, epoch, args.start_epoch, lr, optimizer, model_pos, min_loss) # save best checkpoint

        # For test run, break after one epoch
        try:
            if args.test_run: break
        except:
            pass
##
def train_epoch(args, model_pos, train_loader, losses, optimizer, has_3d, has_gt):
    from hpe_library.my_utils.canonical import batch_rotation_matrix_from_vectors_torch, batch_inverse_rotation_matrices
    model_pos.train()
    pbar = tqdm(train_loader)
    for (batch_input, batch_gt, batch_cam_param) in pbar:
        batch_size = len(batch_input)
        # preprocessing
        batch_gt_original = batch_gt.clone().detach().cuda()
        batch_input, batch_gt, batch_gt_torso, batch_gt_limb, conf = preprocess_train(args, batch_input, batch_gt, has_3d, has_gt)

        # inferece 3D poses
        if args.model in ['DHDSTformer_total', 'DHDSTformer_total2', 'DHDSTformer_total3', 'DHDSTformer_total6', 'DHDSTformer_total7', 'DHDSTformer_total8']:
            predicted_3d_pos, pred_angle, gt_angle = inference_train(args, model_pos, batch_input, batch_gt, batch_gt_torso)
        elif args.model in ['DHDSTformer_total4', 'DHDSTformer_total5']:
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

        # postprocessing
        predicted_3d_pos = postprocess_train(args, predicted_3d_pos, batch_input, batch_gt_original)

        # loss calculation
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
            if args.lambda_canonical_2d_residual > 0:
                if batch_gt.shape[-1] == 3: batch_gt = batch_gt[..., :2]
                if args.canonical_loss == 'mpjpe':
                    loss_canonical_2d_residual = loss_mpjpe(predicted_3d_pos, batch_gt)
                elif args.canonical_loss == 'mse':
                    loss_canonical_2d_residual = nn.MSELoss()(predicted_3d_pos, batch_gt)
                elif args.canonical_loss == 'weighted_mpjpe1':
                    conf = torch.norm(batch_gt - batch_gt[:, :, 0:1, :], dim=-1)
                    loss_canonical_2d_residual = weighted_mpjpe(predicted_3d_pos, batch_gt, conf)
                loss_total += args.lambda_canonical_2d_residual * loss_canonical_2d_residual
                losses['canonical_2d_residual'].update(loss_canonical_2d_residual.item(), batch_size)
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
##
def preprocess_train(args, batch_input, batch_gt, has_3d, has_gt):
    from hpe_library.my_utils.canonical import batch_rotation_matrix_from_vectors_torch, batch_inverse_rotation_matrices
    with torch.no_grad():
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()

        # Train input pre-processing
        if args.no_conf:
            batch_input = batch_input[:, :, :, :2]
        if not has_3d:
            conf = copy.deepcopy(batch_input[:,:,:,2:])    # For 2D data, weight/confidence is at the last channel
        else:
            conf = None
        if args.mask or args.noise: # input augmentation for noise
            batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)
        if args.input_centering:
            batch_input = batch_input - batch_input[:, :, 0:1, :] # root-relative
        if args.norm_input_scale:
            B, F, J, C = batch_input.shape
            scale = torch.norm(batch_input[..., :2].reshape(B, F, 1, 34), dim=-1, keepdim=True)
            batch_input = batch_input / scale

        # Test GT pre-processing
        if args.fix_orientation_gt: # real -> virt
            batch_v_origin_to_pelvis = batch_gt[:, :, 0]
            batch_v_origin_to_pelvis_proj_on_xz = batch_v_origin_to_pelvis.clone()
            batch_v_origin_to_pelvis_proj_on_xz[:, :, 1] = 0
            batch_v_origin_to_principle = torch.tensor([0, 0, 1], device=batch_gt.device).reshape(1, 1, 3).repeat(batch_gt.shape[0], batch_gt.shape[1], 1).float()
            assert batch_v_origin_to_principle.shape == batch_v_origin_to_pelvis.shape, (batch_v_origin_to_principle.shape, batch_v_origin_to_pelvis.shape)
            batch_R1 = batch_rotation_matrix_from_vectors_torch(batch_v_origin_to_pelvis, batch_v_origin_to_pelvis_proj_on_xz)
            batch_R2 = batch_rotation_matrix_from_vectors_torch(batch_v_origin_to_pelvis_proj_on_xz, batch_v_origin_to_principle)
            batch_R_real2virt_from_3d = torch.einsum('bfij,bfjk->bfik', batch_R2, batch_R1)
            batch_R_real2virt_from_3d_inv = torch.linalg.inv(batch_R_real2virt_from_3d)
            batch_gt_virt = torch.einsum('bfij,bfjk->bfik', batch_gt, batch_R_real2virt_from_3d_inv)
            batch_gt = batch_gt_virt

        if args.rootrel: # root-relative 3D pose를 추론하도록 훈련
            batch_gt = batch_gt - batch_gt[:,:,0:1,:] # move the pelvis to the origin for all frames
        else:
            batch_gt[:,:,:,2] = batch_gt[:,:,:,2] - batch_gt[:,0:1,0:1,2] # Place the depth of first frame root to 0. -> 첫번째 프레임의 depth를 0으로 설정
        if args.input_centering:
            batch_gt = batch_gt - batch_gt[:, :, 0:1, :]
        if batch_gt.shape[2] == 17:
            batch_gt_torso = batch_gt[:, :, [0, 1, 4, 7, 8, 9, 10, 11, 14], :]
            batch_gt_limb = batch_gt[:, :, [2, 3, 5, 6, 12, 13, 15, 16], :]
        else:
            batch_gt_torso = None
            batch_gt_limb = None

    return batch_input, batch_gt, batch_gt_torso, batch_gt_limb, conf
##
def inference_train(args, model_pos, batch_input, batch_gt, batch_gt_torso):
    from hpe_library.my_utils.canonical import batch_rotation_matrix_from_vectors_torch, batch_inverse_rotation_matrices
    if args.model in ['DHDSTformer_total', 'DHDSTformer_total2', 'DHDSTformer_total3', 'DHDSTformer_total6', 'DHDSTformer_total7', 'DHDSTformer_total8']:
        predicted_3d_pos = model_pos(batch_input, length_type=args.train_length_type, ref_frame=args.length_frame)
        if args.lambda_dh_angle > 0:
            pred_angle = get_limb_angle(predicted_3d_pos)
            gt_angle = get_limb_angle(batch_gt)
        else:
            pred_angle = None
            gt_angle = None
        return predicted_3d_pos, pred_angle, gt_angle

    elif args.model in ['DHDSTformer_total4', 'DHDSTformer_total5']:
        pred_torso, pred_dh_angle, pred_dh_length, pred_lower_frame_R, pred_upper_frame_R, predicted_3d_pos = model_pos(batch_input)
        return pred_torso, pred_dh_angle, pred_dh_length, pred_lower_frame_R, pred_upper_frame_R, predicted_3d_pos

    elif 'DHDSTformer_limb' in args.model:
        predicted_3d_pos = model_pos(batch_input, batch_gt_torso)
        pred_limb_pos = predicted_3d_pos[:, :, [2, 3, 5, 6, 12, 13, 15, 16], :]
        return predicted_3d_pos, pred_limb_pos

    elif 'DHDSTformer_torso' in args.model:
        # inference
        pred_torso, pred_lower_frame_R, pred_upper_frame_R = model_pos(batch_input)
        return pred_torso, pred_lower_frame_R, pred_upper_frame_R

    elif 'DHDSTformer_torso2' in args.model:
        pred_torso = model_pos(batch_input)
        batch_gt_torso = batch_gt[:, :, [0, 1, 4, 8, 11, 14], :] # r_hip, l_hip, neck, l_shoulder, r_shoulder
        return pred_torso, batch_gt_torso

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
        return pred_3d_pos, gt_3d_pos, pred_root_point, gt_root_point, pred_length, gt_length

    elif 'DHDSTformer_right_arm' == args.model:
        batch_gt_limb = batch_gt[:, :, [14, 15,
                                        16], :]
        pred_limb_pos = model_pos(batch_input)
        return pred_limb_pos, batch_gt_limb

    elif ('DHDSTformer_right_arm2' == args.model) or ('DHDSTformer_right_arm3' == args.model):
        batch_gt_limb = batch_gt[:, :, [0, 14, 15, 16], :]
        pred_limb_pos = model_pos(batch_input)
        return pred_limb_pos, batch_gt_limb

    else:
        predicted_3d_pos = model_pos(batch_input)    # (N, T, 17, 3)
        if args.lambda_dh_angle > 0:
            pred_angle = get_limb_angle(predicted_3d_pos)
            gt_angle = get_limb_angle(batch_gt)
        else:
            pred_angle = None
            gt_angle = None
        return predicted_3d_pos, pred_angle, gt_angle
##
def postprocess_train(args, predicted_3d_pos, batch_input, batch_gt):
    from hpe_library.my_utils.canonical import batch_rotation_matrix_from_vectors_torch
    if args.fix_orientation_pred: # virt -> real
        batch_v_origin_to_pelvis = batch_gt[:, :, 0]
        batch_v_origin_to_pelvis_proj_on_xz = batch_v_origin_to_pelvis.clone()
        batch_v_origin_to_pelvis_proj_on_xz[:, :, 1] = 0
        batch_v_origin_to_principle = torch.tensor([0, 0, 1], device=batch_gt.device).reshape(1, 1, 3).repeat(batch_gt.shape[0], batch_gt.shape[1], 1).float()
        assert batch_v_origin_to_principle.shape == batch_v_origin_to_pelvis.shape, (batch_v_origin_to_principle.shape, batch_v_origin_to_pelvis.shape)
        batch_R1 = batch_rotation_matrix_from_vectors_torch(batch_v_origin_to_pelvis, batch_v_origin_to_pelvis_proj_on_xz)
        batch_R2 = batch_rotation_matrix_from_vectors_torch(batch_v_origin_to_pelvis_proj_on_xz, batch_v_origin_to_principle)
        batch_R_real2virt_from_3d = torch.einsum('bfij,bfjk->bfik', batch_R2, batch_R1)
        batch_R_virt2real_from_3d = torch.linalg.inv(batch_R_real2virt_from_3d)
        batch_R_virt2real_from_3d_inv = batch_R_real2virt_from_3d
        predicted_3d_pos = torch.einsum('bfij,bfjk->bfik', predicted_3d_pos, batch_R_virt2real_from_3d_inv)
    return predicted_3d_pos

##
def generate_loss_dict(args):
    losses = {}
    if args.lambda_3d_pos > 0:                losses['3d_pos'] = AverageMeter()
    if args.lambda_scale > 0:                 losses['3d_scale'] = AverageMeter()
    if args.lambda_3d_velocity > 0:           losses['3d_vel'] = AverageMeter()
    if args.lambda_limb_pos > 0:              losses['3d_pos_limb'] = AverageMeter()
    if args.lambda_limb_scale > 0:            losses['3d_scale_limb'] = AverageMeter()
    if args.lambda_limb_velocity > 0:         losses['3d_vel_limb'] = AverageMeter()
    if args.lambda_torso_pos > 0:             losses['3d_pos_torso'] = AverageMeter()
    if args.lambda_lower_frame_R > 0:         losses['lower_frame_R'] = AverageMeter()
    if args.lambda_upper_frame_R > 0:         losses['upper_frame_R'] = AverageMeter()
    if args.lambda_lg > 0:                    losses['lg'] = AverageMeter()
    if args.lambda_lv > 0:                    losses['lv'] = AverageMeter()
    if args.lambda_a > 0:                     losses['angle'] = AverageMeter()
    if args.lambda_av > 0:                    losses['angle_vel'] = AverageMeter()
    if args.lambda_sym > 0:                   losses['sym'] = AverageMeter()
    if args.lambda_root_point > 0:            losses['root_point'] = AverageMeter()
    if args.lambda_length > 0:                losses['length'] = AverageMeter()
    if args.lambda_dh_angle > 0:              losses['dh_angle'] = AverageMeter()
    if args.lambda_onevec_pos:                losses['onevec_pos'] = AverageMeter()
    if args.lambda_dh_angle2 > 0:             losses['dh_angle2'] = AverageMeter()
    if args.lambda_dh_length > 0:             losses['dh_length'] = AverageMeter()
    if args.lambda_canonical_2d_residual > 0: losses['canonical_2d_residual'] = AverageMeter()
    losses['total'] = AverageMeter()
    losses['2d_proj'] = AverageMeter()
    return losses
##
def update_train_writer(args, train_writer, losses, e1, e2, lr, epoch, total_result_dict, run=None):
    # e1: MPJPE
    # e2: PA-MPJPE
    train_writer.add_scalar('Error P1', e1, epoch + 1)
    train_writer.add_scalar('Error P2', e2, epoch + 1)
    train_writer.add_scalar('lr', lr, epoch + 1)
    train_writer.add_scalar('loss_total', losses['total'].avg, epoch + 1)
    train_writer.add_scalar('loss_2d_proj', losses['2d_proj'].avg, epoch + 1)
    if run is not None: run.log({"Error P1": e1, "Error P2": e2, "lr": lr, "loss_total": losses['total'].avg, "loss_2d_proj": losses['2d_proj'].avg})
    if args.lambda_3d_pos > 0:
        train_writer.add_scalar('loss_3d_pos', losses['3d_pos'].avg, epoch + 1)
        if run is not None: run.log({"loss_3d_pos": losses['3d_pos'].avg})
    if args.lambda_scale > 0:
        train_writer.add_scalar('loss_3d_scale', losses['3d_scale'].avg, epoch + 1)
        if run is not None: run.log({"loss_3d_scale": losses['3d_scale'].avg})
    if args.lambda_3d_velocity > 0:
        train_writer.add_scalar('loss_3d_velocity', losses['3d_vel'].avg, epoch + 1)
        if run is not None: run.log({"loss_3d_velocity": losses['3d_vel'].avg})
    if args.lambda_limb_pos > 0:
        train_writer.add_scalar('loss_3d_pos_limb', losses['3d_pos_limb'].avg, epoch + 1)
        if run is not None: run.log({"loss_3d_pos_limb": losses['3d_pos_limb'].avg})
    if args.lambda_limb_scale > 0:
        train_writer.add_scalar('loss_3d_scale_limb', losses['3d_scale_limb'].avg, epoch + 1)
        if run is not None: run.log({"loss_3d_scale_limb": losses['3d_scale_limb'].avg})
    if args.lambda_limb_velocity > 0:
        train_writer.add_scalar('loss_3d_velocity_limb', losses['3d_vel_limb'].avg, epoch + 1)
        if run is not None: run.log({"loss_3d_velocity_limb": losses['3d_vel_limb'].avg})
    if args.lambda_torso_pos > 0:
        train_writer.add_scalar('loss_3d_pos_torso', losses['3d_pos_torso'].avg, epoch + 1)
        if run is not None: run.log({"loss_3d_pos_torso": losses['3d_pos_torso'].avg})
    if args.lambda_lower_frame_R > 0:
        train_writer.add_scalar('loss_lower_frame_R', losses['lower_frame_R'].avg, epoch + 1)
        if run is not None: run.log({"loss_lower_frame_R": losses['lower_frame_R'].avg})
    if args.lambda_upper_frame_R > 0:
        train_writer.add_scalar('loss_upper_frame_R', losses['upper_frame_R'].avg, epoch + 1)
        if run is not None: run.log({"loss_upper_frame_R": losses['upper_frame_R'].avg})
    if args.lambda_lv > 0:
        train_writer.add_scalar('loss_lv', losses['lv'].avg, epoch + 1)
        if run is not None: run.log({"loss_lv": losses['lv'].avg})
    if args.lambda_lg > 0:
        train_writer.add_scalar('loss_lg', losses['lg'].avg, epoch + 1)
        if run is not None: run.log({"loss_lg": losses['lg'].avg})
    if args.lambda_a > 0:
        train_writer.add_scalar('loss_a', losses['angle'].avg, epoch + 1)
        if run is not None: run.log({"loss_a": losses['angle'].avg})
    if args.lambda_av > 0:
        train_writer.add_scalar('loss_av', losses['angle_vel'].avg, epoch + 1)
        if run is not None: run.log({"loss_av": losses['angle_vel'].avg})
    if args.lambda_sym > 0:
        train_writer.add_scalar('loss_sym', losses['sym'].avg, epoch + 1)
        if run is not None: run.log({"loss_sym": losses['sym'].avg})
    if args.lambda_root_point > 0:
        train_writer.add_scalar('loss_root_point', losses['root_point'].avg, epoch + 1)
        if run is not None: run.log({"loss_root_point": losses['root_point'].avg})
    if args.lambda_length > 0:
        train_writer.add_scalar('loss_length', losses['length'].avg, epoch + 1)
        if run is not None: run.log({"loss_length": losses['length'].avg})
    if args.lambda_dh_angle > 0:
        train_writer.add_scalar('loss_dh_angle', losses['dh_angle'].avg, epoch + 1)
        if run is not None: run.log({"loss_dh_angle": losses['dh_angle'].avg})
    if args.lambda_onevec_pos:
        train_writer.add_scalar('loss_onevec_pos', losses['onevec_pos'].avg, epoch + 1)
        if run is not None: run.log({"loss_onevec_pos": losses['onevec_pos'].avg})
    if args.lambda_dh_angle2 > 0:
        train_writer.add_scalar('loss_dh_angle2', losses['dh_angle2'].avg, epoch + 1)
        if run is not None: run.log({"loss_dh_angle2": losses['dh_angle2'].avg})
    if args.lambda_dh_length > 0:
        train_writer.add_scalar('loss_dh_length', losses['dh_length'].avg, epoch + 1)
        if run is not None: run.log({"loss_dh_length": losses['dh_length'].avg})
    if args.lambda_canonical_2d_residual > 0:
        train_writer.add_scalar('loss_canon_2d_residual', losses['canonical_2d_residual'].avg, epoch + 1)
        if run is not None: run.log({"loss_canon_2d_residual": losses['canonical_2d_residual'].avg})
    if 'arms' in args.part_list:
        arm_mpjpe = np.mean([np.mean(total_result_dict['arms']['results'][key]) for key in total_result_dict['arms']['results'].keys()])
        arm_mpjpe_procrustes = np.mean([np.mean(total_result_dict['arms']['results_procrustes'][key]) for key in total_result_dict['arms']['results_procrustes'].keys()])
        train_writer.add_scalar('arm P1', arm_mpjpe, epoch + 1)
        train_writer.add_scalar('arm P2', arm_mpjpe_procrustes, epoch + 1)
        if run is not None: run.log({"arm P1": arm_mpjpe, "arm P2": arm_mpjpe_procrustes})
    if 'legs' in args.part_list:
        leg_mpjpe = np.mean([np.mean(total_result_dict['legs']['results'][key]) for key in total_result_dict['legs']['results'].keys()])
        leg_mpjpe_procrustes = np.mean([np.mean(total_result_dict['legs']['results_procrustes'][key]) for key in total_result_dict['legs']['results_procrustes'].keys()])
        train_writer.add_scalar('leg P1', leg_mpjpe, epoch + 1)
        train_writer.add_scalar('leg P2', leg_mpjpe_procrustes, epoch + 1)
        if run is not None: run.log({"leg P1": leg_mpjpe, "leg P2": leg_mpjpe_procrustes})

    return train_writer