import sys
import copy
import torch
import numpy as np
import torch.nn as nn
sys.path.append('/home/hrai/codes/hpe_library')
from my_utils import get_limb_angle, get_batch_lower_torso_frame_from_pose, get_batch_upper_torso_frame_from_pose, matrix_to_quaternion, batch_azim_elev_to_vec, get_input_gt_for_onevec

def preprocess_train(args, batch_input, batch_gt, has_3d, has_gt):
    with torch.no_grad():
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
        if args.no_conf:
            batch_input = batch_input[:, :, :, :2]
        if not has_3d:
            conf = copy.deepcopy(batch_input[:,:,:,2:])    # For 2D data, weight/confidence is at the last channel
        else:
            conf = None
        if args.rootrel:
            batch_gt = batch_gt - batch_gt[:,:,0:1,:] # move the pelvis to the origin
        else:
            batch_gt[:,:,:,2] = batch_gt[:,:,:,2] - batch_gt[:,0:1,0:1,2] # Place the depth of first frame root to 0. -> 첫번째 프레임의 depth를 0으로 설정
        if args.mask or args.noise:
            batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)
        if args.canonical:
            batch_input = batch_input - batch_input[:, :, 0:1, :] # root-relative
            batch_gt = batch_gt - batch_gt[:, :, 0:1, :]
        if batch_gt.shape[2] == 17:
            batch_gt_torso = batch_gt[:, :, [0, 1, 4, 7, 8, 9, 10, 11, 14], :]
            batch_gt_limb = batch_gt[:, :, [2, 3, 5, 6, 12, 13, 15, 16], :]
        else:
            batch_gt_torso = None
            batch_gt_limb = None
    
    return batch_input, batch_gt, batch_gt_torso, batch_gt_limb, conf

def inference_train(args, model_pos, batch_input, batch_gt, batch_gt_torso):
    if args.model in ['DHDSTformer_total', 'DHDSTformer_total2', 'DHDSTformer_total3', 'DHDSTformer_total6']: 
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
        batch_gt_limb = batch_gt[:, :, [14, 15, 16], :]
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