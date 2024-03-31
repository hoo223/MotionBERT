import torch
from tqdm import tqdm
import numpy as np
import prettytable
from lib.model.loss import *
from lib.utils.utils_data import flip_data


def preprocess_eval(args, batch_input, batch_gt):
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
    else:
        batch_gt_torso = None
        batch_gt_limb_pos = None
    return batch_input, batch_gt, batch_gt_torso, batch_gt_limb_pos

def batch_inference_eval(args, model_pos, batch_input, batch_gt_torso, batch_gt_limb_pos):
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
            
    return predicted_3d_pos

def inference_eval(args, model_pos, test_loader, datareader, denormalize=True):
    results_all = []
    gts_all = []
    inputs_all = []
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader): # batch_input: normalized joint_2d, batch_gt: normalized joint3d_image
            batch_size = len(batch_input) 
            # preprocessing
            batch_input, batch_gt, batch_gt_torso, batch_gt_limb_pos = preprocess_eval(args, batch_input, batch_gt)
            # inference
            predicted_3d_pos = batch_inference_eval(args, model_pos, batch_input, batch_gt_torso, batch_gt_limb_pos)
            # postprocessing
            if args.rootrel:
                predicted_3d_pos[:,:,0,:] = 0     # [N,T,17,3]
            else:
                batch_gt[:,0,0,2] = 0
            if args.gt_2d: # input 2d를 추론값으로 사용함으로써 depth만 추정하도록 함
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            # store the results
            results_all.append(predicted_3d_pos.cpu().numpy())
            gts_all.append(batch_gt.cpu().numpy())
            inputs_all.append(batch_input.cpu().numpy())
    results_all = np.concatenate(results_all)
    gts_all = np.concatenate(gts_all)
    inputs_all = np.concatenate(inputs_all)
    if denormalize:
        results_all = datareader.denormalize(results_all) # denormalize the predicted 3D poses
    
    return results_all

def get_clip_info(datareader, results_all):
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

    return num_test_frames, action_clips, factor_clips, source_clips, frame_clips, gt_clips

def calculate_eval_metric(args, results_all, datareader):
    num_test_frames, action_clips, factor_clips, source_clips, frame_clips, gt_clips = get_clip_info(datareader, results_all)

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
    return e1, e2, total_result_dict


# ------------------------------------------------------------------------------------------- # 
# for inference of onevec model

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