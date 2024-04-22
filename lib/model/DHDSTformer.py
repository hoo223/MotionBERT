## 함수화 하기
import sys
sys.path.append('/home/hrai/codes/hpe_library')
from lib_import import *
from my_utils import *

from functools import partial
from lib.model.DSTformer import DSTformer
from lib.utils.learning import * # load_backbone
import torch.nn.functional as F

class DHDSTformer_total(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        # appendage id
        self.right_arm_id = 0
        self.left_arm_id  = 1
        self.right_leg_id = 2
        self.left_leg_id  = 3
        
        self.dstformer_backbone = load_backbone(args)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        self.dstformer_backbone = self.dstformer_backbone
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        
        self.torso_head = nn.Linear(args.dim_rep, dim_out)
        self.right_arm_head1 = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.right_arm_head2 = nn.Linear(args.dim_rep, 6) # 6: yaw1, pitch1, yaw2, pitch2, length1, length2
        self.left_arm_head1  = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.left_arm_head2  = nn.Linear(args.dim_rep, 6) # 6: yaw1, pitch1, yaw2, pitch2, length1, length2
        self.right_leg_head1 = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.right_leg_head2 = nn.Linear(args.dim_rep, 6) # 6: yaw1, pitch1, yaw2, pitch2, length1, length2
        self.left_leg_head1  = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.left_leg_head2  = nn.Linear(args.dim_rep, 6) # 6: yaw1, pitch1, yaw2, pitch2, length1, length2
        
    def forward(self, batch_input, length_type='each', ref_frame=0):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation
        
        # update batch size
        self.batch_size = batch_input.shape[0]
        self.batch_dh_model = BatchDHModel(None, batch_size=self.batch_size, num_frames=self.num_frames, data_type=self.data_type, device=batch_input.device)
        self.batch_dh_model.batch_size = self.batch_size

        # inference
        rep = self.dstformer_backbone.module.get_representation(batch_input)
        torso_output     = self.torso_head(rep[:, :, [0, 1, 4, 7, 8, 9, 10, 11, 14], :]) # (N, F, 9, 3)
        right_arm_output = self.right_arm_head2(self.right_arm_head1(rep[:, :, [14, 15, 16], :].flatten(2))) # (N, F, 6)
        left_arm_output  =  self.left_arm_head2(self.left_arm_head1(rep[:, :, [11, 12, 13], :].flatten(2))) # (N, F, 6)
        right_leg_output = self.right_leg_head2(self.right_leg_head1(rep[:, :, [1, 2, 3], :].flatten(2))) # (N, F, 6)
        left_leg_output  =  self.left_leg_head2(self.left_leg_head1(rep[:, :, [4, 5, 6], :].flatten(2))) # (N, F, 6)
        
        # update dh model
        self.batch_dh_model.set_batch_dh_model_from_dhdst_output(torso_output, right_arm_output, left_arm_output, right_leg_output, left_leg_output)
        
        return self.batch_dh_model.get_batch_pose_3d()
    
class DHDSTformer_total2(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        # appendage id
        self.right_arm_id = 0
        self.left_arm_id  = 1
        self.right_leg_id = 2
        self.left_leg_id  = 3
        
        self.dstformer_backbone = load_backbone(args)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        self.dstformer_backbone = self.dstformer_backbone
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        
        self.torso_head = nn.Linear(args.dim_rep, dim_out)
        self.right_arm_head = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.right_arm_head_angle = nn.Linear(args.dim_rep, 4) # 4: yaw1, pitch1, yaw2, pitch2
        self.right_arm_head_length = nn.Linear(args.dim_rep, 2) # 4: length1, length2
        self.left_arm_head  = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.left_arm_head_angle = nn.Linear(args.dim_rep, 4) # 4: yaw1, pitch1, yaw2, pitch2
        self.left_arm_head_length = nn.Linear(args.dim_rep, 2) # 4: length1, length2
        self.right_leg_head = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.right_leg_head_angle = nn.Linear(args.dim_rep, 4) # 4: yaw1, pitch1, yaw2, pitch2
        self.right_leg_head_length = nn.Linear(args.dim_rep, 2) # 4: length1, length2
        self.left_leg_head  = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.left_leg_head_angle = nn.Linear(args.dim_rep, 4) # 4: yaw1, pitch1, yaw2, pitch2
        self.left_leg_head_length = nn.Linear(args.dim_rep, 2) # 4: length1, length2
         
    def forward(self, batch_input, length_type='each', ref_frame=0):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation
        
        # update batch size
        self.batch_size = batch_input.shape[0]
        self.batch_dh_model = BatchDHModel(None, batch_size=self.batch_size, num_frames=self.num_frames, data_type=self.data_type, device=batch_input.device)
        self.batch_dh_model.batch_size = self.batch_size

        # inference
        rep = self.dstformer_backbone.module.get_representation(batch_input)
        torso_output  = self.torso_head(rep[:, :, [0, 1, 4, 7, 8, 9, 10, 11, 14], :]) # (N, F, 9, 3)
        right_arm_rep = self.right_arm_head(rep[:, :, [14, 15, 16], :].flatten(2))
        left_arm_rep  =  self.left_arm_head(rep[:, :, [11, 12, 13], :].flatten(2))
        right_leg_rep = self.right_leg_head(rep[:, :, [1, 2, 3], :].flatten(2))
        left_leg_rep  =  self.left_leg_head(rep[:, :, [4, 5, 6], :].flatten(2))
         
        right_arm_angle  = self.right_arm_head_angle(right_arm_rep) # (N, F, 4)
        left_arm_angle   = self.left_arm_head_angle(left_arm_rep) # (N, F, 4)
        right_leg_angle  = self.right_leg_head_angle(right_leg_rep) # (N, F, 4)
        left_leg_angle   = self.left_leg_head_angle(left_leg_rep) # (N, F, 4)
        right_arm_length = self.right_arm_head_length(right_arm_rep) # (N, F, 2)
        left_arm_length  = self.left_arm_head_length(left_arm_rep) # (N, F, 2)
        right_leg_length = self.right_leg_head_length(right_leg_rep) # (N, F, 2)
        left_leg_length  = self.left_leg_head_length(left_leg_rep) # (N, F, 2)
        
        right_arm_output = torch.cat([right_arm_angle, right_arm_length], dim=-1)
        left_arm_output  = torch.cat([left_arm_angle, left_arm_length], dim=-1)
        right_leg_output = torch.cat([right_leg_angle, right_leg_length], dim=-1)
        left_leg_output  = torch.cat([left_leg_angle, left_leg_length], dim=-1)
        
        # update dh model
        self.batch_dh_model.set_batch_dh_model_from_dhdst_output(torso_output, right_arm_output, left_arm_output, right_leg_output, left_leg_output)
        
        return self.batch_dh_model.get_batch_pose_3d()
    
class DHDSTformer_total3(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        # appendage id
        self.right_arm_id = 0
        self.left_arm_id  = 1
        self.right_leg_id = 2
        self.left_leg_id  = 3
        
        self.dstformer_backbone = load_backbone(args)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            
        num_keypoints = 17
        
        self.torso_head            = nn.Linear(args.dim_rep, dim_out)
        self.flatten_head          = nn.Linear(args.dim_rep*num_keypoints, args.dim_rep*3)
        self.right_arm_head        = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.right_arm_head_angle  = nn.Linear(args.dim_rep, 4) # 4: yaw1, pitch1, yaw2, pitch2
        self.right_arm_head_length = nn.Linear(args.dim_rep, 2) # 4: length1, length2
        self.left_arm_head         = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.left_arm_head_angle   = nn.Linear(args.dim_rep, 4) # 4: yaw1, pitch1, yaw2, pitch2
        self.left_arm_head_length  = nn.Linear(args.dim_rep, 2) # 4: length1, length2
        self.right_leg_head        = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.right_leg_head_angle  = nn.Linear(args.dim_rep, 4) # 4: yaw1, pitch1, yaw2, pitch2
        self.right_leg_head_length = nn.Linear(args.dim_rep, 2) # 4: length1, length2
        self.left_leg_head         = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.left_leg_head_angle   = nn.Linear(args.dim_rep, 4) # 4: yaw1, pitch1, yaw2, pitch2
        self.left_leg_head_length  = nn.Linear(args.dim_rep, 2) # 4: length1, length2
         
    def forward(self, batch_input, length_type='each', ref_frame=0):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation
        
        # update batch size
        self.batch_size = batch_input.shape[0]
        self.batch_dh_model = BatchDHModel(None, batch_size=self.batch_size, num_frames=self.num_frames, data_type=self.data_type, device=batch_input.device)
        self.batch_dh_model.batch_size = self.batch_size

        # inference
        rep = self.dstformer_backbone.module.get_representation(batch_input)
        torso_output  = self.torso_head(rep[:, :, [0, 1, 4, 7, 8, 9, 10, 11, 14], :])
        rep_flatten   = self.flatten_head(rep.flatten(2)) # (N, F, 17*dim_re) -> (N, F, 3*dim_rep)
        right_arm_rep = self.right_arm_head(rep_flatten)  # (N, F, 3*dim_rep) -> (N, F, dim_rep)
        left_arm_rep  = self.left_arm_head(rep_flatten)   # (N, F, 3*dim_rep) -> (N, F, dim_rep)
        right_leg_rep = self.right_leg_head(rep_flatten)  # (N, F, 3*dim_rep) -> (N, F, dim_rep)
        left_leg_rep  = self.left_leg_head(rep_flatten)   # (N, F, 3*dim_rep) -> (N, F, dim_rep)
         
        right_arm_angle  = self.right_arm_head_angle(right_arm_rep) # (N, F, 17, dim_rep) -> (N, F, 4)
        right_arm_length = self.right_arm_head_length(right_arm_rep) # (N, F, 2)
        
        left_arm_angle   = self.left_arm_head_angle(left_arm_rep) # (N, F, 4)
        left_arm_length  = self.left_arm_head_length(left_arm_rep) # (N, F, 2)
        
        right_leg_angle  = self.right_leg_head_angle(right_leg_rep) # (N, F, 4)
        right_leg_length = self.right_leg_head_length(right_leg_rep) # (N, F, 2)
        
        left_leg_angle   = self.left_leg_head_angle(left_leg_rep) # (N, F, 4)        
        left_leg_length  = self.left_leg_head_length(left_leg_rep) # (N, F, 2)
        
        right_arm_output = torch.cat([right_arm_angle, right_arm_length], dim=-1)
        left_arm_output  = torch.cat([left_arm_angle, left_arm_length], dim=-1)
        right_leg_output = torch.cat([right_leg_angle, right_leg_length], dim=-1)
        left_leg_output  = torch.cat([left_leg_angle, left_leg_length], dim=-1)
        
        # update dh model
        self.batch_dh_model.set_batch_dh_model_from_dhdst_output(torso_output, right_arm_output, left_arm_output, right_leg_output, left_leg_output)
        
        return self.batch_dh_model.get_batch_pose_3d()
    
class DHDSTformer_total4(nn.Module):
    def __init__(self, args, dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        self.torso_keypoints = [0, 1, 4, 7, 8, 9, 10, 11, 14]
        self.right_arm_keypoints = [14, 15, 16]
        self.left_arm_keypoints = [11, 12, 13]
        self.arm_keypoints = self.right_arm_keypoints + self.left_arm_keypoints
        self.right_leg_keypoints = [1, 2, 3]
        self.left_leg_keypoints = [4, 5, 6]
        self.leg_keypoints = self.right_leg_keypoints + self.left_leg_keypoints

        # backbone
        self.dstformer_backbone = load_backbone(args)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        self.torso_head = nn.Linear(args.dim_rep, 3)
        self.angle_head = nn.Linear(args.dim_rep, 4) # 4: yaw1, pitch1, yaw2, pitch2
        self.length_head = nn.Linear(args.dim_rep, 2) # 4: length1, length2

        # batch_dh_model
        self.batch_dh_model = BatchDHModel(None, batch_size=self.batch_size, num_frames=self.num_frames, length_type='first')
         
    def forward(self, batch_input):
        # batch_x: (B, F, 17, 2) 2d pose

        # update batch size
        self.batch_size = batch_input.shape[0]
        self.batch_dh_model = BatchDHModel(None, batch_size=self.batch_size, num_frames=self.num_frames, length_type='first')
        self.batch_dh_model.batch_size = self.batch_size

        # feature extraction
        feature = self.dstformer_backbone.module.get_representation(batch_input) # [B, F, J, dim_rep]
        torso_feature = feature[:, :, self.torso_keypoints, :] # (B, F, 9, dim_rep)
        right_arm_feature = feature[:, :, self.right_arm_keypoints, :] # (B, F, 3, dim_rep)
        left_arm_feature  = feature[:, :, self.left_arm_keypoints, :]  # (B, F, 3, dim_rep)
        arm_feature = feature[:, :, self.arm_keypoints, :]  # (B, F, 6, dim_rep)
        right_leg_feature = feature[:, :, self.right_leg_keypoints, :] # (B, F, 3, dim_rep)
        left_leg_feature  = feature[:, :, self.left_leg_keypoints, :]  # (B, F, 3, dim_rep)
        leg_feature = feature[:, :, self.leg_keypoints, :]  # (B, F, 6, dim_rep)

        # torso output
        torso_output  = self.torso_head(torso_feature) # (B, F, 9, 3)
        batch_pelvis = torso_output[:, :, 0, :]
        batch_r_hip = torso_output[:, :, 1, :]
        batch_l_hip = torso_output[:, :, 2, :]
        batch_torso = torso_output[:, :, 3, :]
        batch_neck = torso_output[:, :, 4, :]
        #batch_nose = torso_output[:, :, 5, :]
        #batch_head = torso_output[:, :, 6, :]
        batch_l_shoulder = torso_output[:, :, 7, :]
        batch_r_shoulder = torso_output[:, :, 8, :]
        batch_lower_frame_origin, batch_lower_frame_R = get_batch_lower_torso_frame_from_keypoints(batch_r_hip, batch_l_hip, batch_pelvis, batch_torso)
        batch_upper_frame_origin, batch_upper_frame_R = get_batch_upper_torso_frame_from_keypoints(batch_r_shoulder, batch_l_shoulder, batch_torso, batch_neck)

        # limb output -> left/right length should be same for upper/lower arm/leg
        right_arm_angle = F.adaptive_avg_pool2d(self.angle_head(right_arm_feature), (1, 4)).squeeze(2)  # (B, F, 4) -> 0: upper azim, 1: upper elev, 2: lower azim, 3: lower elev
        left_arm_angle  = F.adaptive_avg_pool2d(self.angle_head(left_arm_feature), (1, 4)).squeeze(2)   # (B, F, 4)
        right_leg_angle = F.adaptive_avg_pool2d(self.angle_head(right_leg_feature), (1, 4)) .squeeze(2) # (B, F, 4)
        left_leg_angle  = F.adaptive_avg_pool2d(self.angle_head(left_leg_feature), (1, 4)).squeeze(2)   # (B, F, 4)
        angle_output = torch.cat([right_arm_angle, left_arm_angle, right_leg_angle, left_leg_angle], dim=-1) # (B, F, 16)

        right_arm_length = F.adaptive_avg_pool2d(self.length_head(right_arm_feature).permute(0, 3, 2, 1), (1, 1)).squeeze(2) # (B, 2, 1) -> 0: upper length, 1: lower length
        left_arm_length  = F.adaptive_avg_pool2d(self.length_head(left_arm_feature).permute(0, 3, 2, 1), (1, 1)).squeeze(2)  # (B, 2, 1)
        right_leg_length = F.adaptive_avg_pool2d(self.length_head(right_leg_feature).permute(0, 3, 2, 1), (1, 1)).squeeze(2) # (B, 2, 1)
        left_leg_length  = F.adaptive_avg_pool2d(self.length_head(left_leg_feature).permute(0, 3, 2, 1), (1, 1)).squeeze(2)  # (B, 2, 1)
        length_output = torch.cat([right_arm_length, left_arm_length, right_leg_length, left_leg_length], dim=-1) # (B, 2, 4)
        
        # update dh model
        self.batch_dh_model.set_batch_torso(torso_output)
        self.batch_dh_model.set_batch_torso_frame()
        self.batch_dh_model.set_batch_length(length_output, update_appendage=False)
        self.batch_dh_model.set_batch_angle(angle_output, update_appendage=False)
        self.batch_dh_model.generate_all_batch_appendages()
        self.batch_dh_model.forward_batch_appendage()
        predicted_3d_pos = self.batch_dh_model.get_batch_pose_3d()

        return torso_output, angle_output, length_output, batch_lower_frame_R, batch_upper_frame_R, predicted_3d_pos
    
class linear_head(nn.Module):
    def __init__(self, linear_size=1024, out_dim=3, num_layers=2, p_dropout=0.5):
        super(linear_head, self).__init__()

        self.linear_stages = nn.ModuleList()
        for i in range(num_layers):
            self.linear_stages.append(nn.Linear(linear_size, linear_size))
            self.linear_stages.append(nn.ReLU())
            self.linear_stages.append(nn.Dropout(p_dropout))            
        self.linear_stages.append(nn.Linear(linear_size, int(linear_size/2)))
        self.linear_stages.append(nn.Linear(int(linear_size/2), out_dim))

    def forward(self, x):
        for linear in self.linear_stages:
            x = linear(x)
        return x
    
class DHDSTformer_total5(nn.Module):
    def __init__(self, args, dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        self.torso_keypoints = [0, 1, 4, 7, 8, 9, 10, 11, 14]
        self.right_arm_keypoints = [14, 15, 16]
        self.left_arm_keypoints = [11, 12, 13]
        self.arm_keypoints = self.right_arm_keypoints + self.left_arm_keypoints
        self.right_leg_keypoints = [1, 2, 3]
        self.left_leg_keypoints = [4, 5, 6]
        self.leg_keypoints = self.right_leg_keypoints + self.left_leg_keypoints

        # backbone
        self.dstformer_backbone = load_backbone(args)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        self.torso_head = linear_head(linear_size=args.dim_rep, out_dim=3)
        self.angle_head = linear_head(linear_size=args.dim_rep, out_dim=4) # 4: yaw1, pitch1, yaw2, pitch2
        self.length_head = linear_head(linear_size=args.dim_rep, out_dim=2) # 4: length1, length2

        # batch_dh_model
        self.batch_dh_model = BatchDHModel(None, batch_size=self.batch_size, num_frames=self.num_frames, length_type='first')
         
    def forward(self, batch_input):
        # batch_x: (B, F, 17, 2) 2d pose

        # update batch size
        self.batch_size = batch_input.shape[0]
        self.batch_dh_model = BatchDHModel(None, batch_size=self.batch_size, num_frames=self.num_frames, length_type='first')
        self.batch_dh_model.batch_size = self.batch_size

        # feature extraction
        feature = self.dstformer_backbone.module.get_representation(batch_input) # [B, F, J, dim_rep]
        torso_feature = feature[:, :, self.torso_keypoints, :] # (B, F, 9, dim_rep)
        right_arm_feature = feature[:, :, self.right_arm_keypoints, :] # (B, F, 3, dim_rep)
        left_arm_feature  = feature[:, :, self.left_arm_keypoints, :]  # (B, F, 3, dim_rep)
        arm_feature = feature[:, :, self.arm_keypoints, :]  # (B, F, 6, dim_rep)
        right_leg_feature = feature[:, :, self.right_leg_keypoints, :] # (B, F, 3, dim_rep)
        left_leg_feature  = feature[:, :, self.left_leg_keypoints, :]  # (B, F, 3, dim_rep)
        leg_feature = feature[:, :, self.leg_keypoints, :]  # (B, F, 6, dim_rep)

        # torso output
        torso_output  = self.torso_head(torso_feature) # (B, F, 9, 3)
        batch_pelvis = torso_output[:, :, 0, :]
        batch_r_hip = torso_output[:, :, 1, :]
        batch_l_hip = torso_output[:, :, 2, :]
        batch_torso = torso_output[:, :, 3, :]
        batch_neck = torso_output[:, :, 4, :]
        #batch_nose = torso_output[:, :, 5, :]
        #batch_head = torso_output[:, :, 6, :]
        batch_l_shoulder = torso_output[:, :, 7, :]
        batch_r_shoulder = torso_output[:, :, 8, :]
        batch_lower_frame_origin, batch_lower_frame_R = get_batch_lower_torso_frame_from_keypoints(batch_r_hip, batch_l_hip, batch_pelvis, batch_torso)
        batch_upper_frame_origin, batch_upper_frame_R = get_batch_upper_torso_frame_from_keypoints(batch_r_shoulder, batch_l_shoulder, batch_torso, batch_neck)

        # limb output -> left/right length should be same for upper/lower arm/leg
        right_arm_angle = F.adaptive_avg_pool2d(self.angle_head(right_arm_feature), (1, 4)).squeeze(2)  # (B, F, 4) -> 0: upper azim, 1: upper elev, 2: lower azim, 3: lower elev
        left_arm_angle  = F.adaptive_avg_pool2d(self.angle_head(left_arm_feature), (1, 4)).squeeze(2)   # (B, F, 4)
        right_leg_angle = F.adaptive_avg_pool2d(self.angle_head(right_leg_feature), (1, 4)) .squeeze(2) # (B, F, 4)
        left_leg_angle  = F.adaptive_avg_pool2d(self.angle_head(left_leg_feature), (1, 4)).squeeze(2)   # (B, F, 4)
        angle_output = torch.cat([right_arm_angle, left_arm_angle, right_leg_angle, left_leg_angle], dim=-1) # (B, F, 16)

        right_arm_length = F.adaptive_avg_pool2d(self.length_head(right_arm_feature).permute(0, 3, 2, 1), (1, 1)).squeeze(2) # (B, 2, 1) -> 0: upper length, 1: lower length
        left_arm_length  = F.adaptive_avg_pool2d(self.length_head(left_arm_feature).permute(0, 3, 2, 1), (1, 1)).squeeze(2)  # (B, 2, 1)
        right_leg_length = F.adaptive_avg_pool2d(self.length_head(right_leg_feature).permute(0, 3, 2, 1), (1, 1)).squeeze(2) # (B, 2, 1)
        left_leg_length  = F.adaptive_avg_pool2d(self.length_head(left_leg_feature).permute(0, 3, 2, 1), (1, 1)).squeeze(2)  # (B, 2, 1)
        length_output = torch.cat([right_arm_length, left_arm_length, right_leg_length, left_leg_length], dim=-1) # (B, 2, 4)
        
        # update dh model
        self.batch_dh_model.set_batch_torso(torso_output)
        self.batch_dh_model.set_batch_torso_frame()
        self.batch_dh_model.set_batch_length(length_output, update_appendage=False)
        self.batch_dh_model.set_batch_angle(angle_output, update_appendage=False)
        self.batch_dh_model.generate_all_batch_appendages()
        self.batch_dh_model.forward_batch_appendage()
        predicted_3d_pos = self.batch_dh_model.get_batch_pose_3d()

        return torso_output, angle_output, length_output, batch_lower_frame_R, batch_upper_frame_R, predicted_3d_pos
    
# 3D pose -> IK by neural network -> FK -> MPJPE
class DHDSTformer_total6(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, num_layers_head=1, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        # appendage id
        self.right_arm_id = 0
        self.left_arm_id  = 1
        self.right_leg_id = 2
        self.left_leg_id  = 3
        
        self.dstformer_backbone = load_backbone(args)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        
        #self.torso_head = nn.Linear(args.dim_rep, dim_out)
        self.right_arm_head = nn.Linear(dim_out*3, args.dim_rep)
        self.right_arm_head_angle = linear_head(linear_size=args.dim_rep, num_layers=num_layers_head, out_dim=4) # 4: yaw1, pitch1, yaw2, pitch2
        self.right_arm_head_length = linear_head(linear_size=args.dim_rep, num_layers=num_layers_head, out_dim=2) # 2: length1, length2
        self.left_arm_head  = nn.Linear(dim_out*3, args.dim_rep)
        self.left_arm_head_angle = linear_head(linear_size=args.dim_rep, num_layers=num_layers_head, out_dim=4) # 4: yaw1, pitch1, yaw2, pitch2
        self.left_arm_head_length = linear_head(linear_size=args.dim_rep, num_layers=num_layers_head, out_dim=2) # 2: length1, length2
        self.right_leg_head = nn.Linear(dim_out*3, args.dim_rep)
        self.right_leg_head_angle = linear_head(linear_size=args.dim_rep, num_layers=num_layers_head, out_dim=4) # 4: yaw1, pitch1, yaw2, pitch2
        self.right_leg_head_length = linear_head(linear_size=args.dim_rep, num_layers=num_layers_head, out_dim=2) # 2: length1, length2
        self.left_leg_head  = nn.Linear(dim_out*3, args.dim_rep)
        self.left_leg_head_angle = linear_head(linear_size=args.dim_rep, num_layers=num_layers_head, out_dim=4) # 4: yaw1, pitch1, yaw2, pitch2
        self.left_leg_head_length = linear_head(linear_size=args.dim_rep, num_layers=num_layers_head, out_dim=2) # 2: length1, length2
         
    def forward(self, batch_input, length_type='each', ref_frame=0):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation
        
        # update batch size
        self.batch_size = batch_input.shape[0]
        self.batch_dh_model = BatchDHModel(None, batch_size=self.batch_size, num_frames=self.num_frames, data_type=self.data_type, device=batch_input.device)
        self.batch_dh_model.batch_size = self.batch_size

        # inference
        rep = self.dstformer_backbone.module.forward(batch_input)
        torso_output  = rep[:, :, [0, 1, 4, 7, 8, 9, 10, 11, 14], :] # (N, F, 9, 3)
        right_arm_rep = self.right_arm_head(rep[:, :, [14, 15, 16], :].flatten(2)) # (N, F, 9) -> (N, F, dim_rep)
        left_arm_rep  =  self.left_arm_head(rep[:, :, [11, 12, 13], :].flatten(2))
        right_leg_rep = self.right_leg_head(rep[:, :, [1, 2, 3], :].flatten(2))
        left_leg_rep  =  self.left_leg_head(rep[:, :, [4, 5, 6], :].flatten(2))
         
        right_arm_angle  = self.right_arm_head_angle(right_arm_rep) # (N, F, dim_rep) -> (N, F, 4)
        left_arm_angle   = self.left_arm_head_angle(left_arm_rep) # (N, F, dim_rep) -> (N, F, 4)
        right_leg_angle  = self.right_leg_head_angle(right_leg_rep) # (N, F, dim_rep) -> (N, F, 4)
        left_leg_angle   = self.left_leg_head_angle(left_leg_rep) # (N, F, dim_rep) -> (N, F, 4)
        right_arm_length = self.right_arm_head_length(right_arm_rep) # (N, F, dim_rep) -> (N, F, 2)
        left_arm_length  = self.left_arm_head_length(left_arm_rep) # (N, F, dim_rep) -> (N, F, 2)
        right_leg_length = self.right_leg_head_length(right_leg_rep) # (N, F, dim_rep) -> (N, F, 2)
        left_leg_length  = self.left_leg_head_length(left_leg_rep) # (N, F, dim_rep) -> (N, F, 2)
        
        right_arm_output = torch.cat([right_arm_angle, right_arm_length], dim=-1)
        left_arm_output  = torch.cat([left_arm_angle, left_arm_length], dim=-1)
        right_leg_output = torch.cat([right_leg_angle, right_leg_length], dim=-1)
        left_leg_output  = torch.cat([left_leg_angle, left_leg_length], dim=-1)
        
        # update dh model
        self.batch_dh_model.set_batch_dh_model_from_dhdst_output(torso_output, right_arm_output, left_arm_output, right_leg_output, left_leg_output)
        
        return self.batch_dh_model.get_batch_pose_3d()
    
class DHDSTformer_torso(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        self.dstformer_backbone = load_backbone(args)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        self.torso_head    = nn.Linear(args.dim_rep, dim_out)
         
    def forward(self, batch_input):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation

        # inference
        rep = self.dstformer_backbone.module.get_representation(batch_input)
        torso_output = F.adaptive_avg_pool2d(rep, (9, 3)) # self.torso_head(rep[:, :, [0, 1, 4, 7, 8, 9, 10, 11, 14], :]) # [B, F, 9, 3] # torso_full
        batch_pelvis = torso_output[:, :, 0, :]
        batch_r_hip = torso_output[:, :, 1, :]
        batch_l_hip = torso_output[:, :, 2, :]
        batch_torso = torso_output[:, :, 3, :]
        batch_neck = torso_output[:, :, 4, :]
        batch_nose = torso_output[:, :, 5, :]
        batch_head = torso_output[:, :, 6, :]
        batch_l_shoulder = torso_output[:, :, 7, :]
        batch_r_shoulder = torso_output[:, :, 8, :]

        batch_lower_frame_origin, batch_lower_frame_R = get_batch_lower_torso_frame_from_keypoints(batch_r_hip, batch_l_hip, batch_pelvis, batch_torso)
        batch_upper_frame_origin, batch_upper_frame_R = get_batch_upper_torso_frame_from_keypoints(batch_r_shoulder, batch_l_shoulder, batch_torso, batch_neck)
        
        return torso_output, batch_lower_frame_R, batch_upper_frame_R
    
class DHDSTformer_torso2(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        self.dstformer_backbone = DSTformer(dim_in=3, dim_out=3, dim_feat=args.dim_feat, dim_rep=args.dim_rep, 
                                   depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                   maxlen=args.maxlen, num_joints=args.num_joints)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=False)
         
    def forward(self, batch_input):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation

        # inference
        rep = self.dstformer_backbone.module.get_representation(batch_input)
        torso_output = F.adaptive_avg_pool2d(rep, (6, 3)) # torso_small
        # batch_pelvis = torso_output[:, :, 0, :]
        # batch_r_hip = torso_output[:, :, 1, :]
        # batch_l_hip = torso_output[:, :, 2, :]
        # batch_neck = torso_output[:, :, 3, :]
        # batch_l_shoulder = torso_output[:, :, 4, :]
        # batch_r_shoulder = torso_output[:, :, 5, :]
        
        return torso_output

class DHDSTformer_onevec(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        self.dstformer_backbone = DSTformer(dim_in=2, dim_out=3, dim_feat=args.dim_feat, dim_rep=args.dim_rep, 
                                   depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                   maxlen=args.maxlen, num_joints=args.num_joints)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=False)
        self.root_head   = nn.Linear(args.dim_rep*args.num_joints, 3)
        self.length_head = nn.Linear(args.maxlen, 1)
        self.angle_head  = nn.Linear(args.dim_rep*args.num_joints, 2)
         
    def forward(self, batch_input):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation

        # inference
        rep = self.dstformer_backbone.module.get_representation(batch_input) # [B, F, J, dim_rep]
        rep_flatten = rep.flatten(2) # [B, F, J*dim_rep]
        rep_length = F.adaptive_avg_pool2d(rep, (1, 1)).flatten(1) # [B, F]
        root_traj = self.root_head(rep_flatten) # [B, F, 3]
        length = self.length_head(rep_length) # [B, 1]
        angle = self.angle_head(rep_flatten) # [B, F, 2]
        
        return root_traj, length, angle


class DHDSTformer_torso_limb(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        self.dstformer_backbone_torso = load_backbone(args)
        self.dstformer_backbone_torso = nn.DataParallel(self.dstformer_backbone_torso)
        self.dstformer_backbone_limb = load_backbone(args)
        self.dstformer_backbone_limb = nn.DataParallel(self.dstformer_backbone_limb)
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone_torso.load_state_dict(checkpoint['model_pos'], strict=True)
            self.dstformer_backbone_limb.load_state_dict(checkpoint['model_pos'], strict=True)

    def forward(self, batch_input):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation

        # inference
        rep_torso = self.dstformer_backbone_torso.module.get_representation(batch_input)
        rep_limb = self.dstformer_backbone_limb.module.get_representation(batch_input)
        torso_output = F.adaptive_avg_pool2d(rep_torso, (9, 3)) # self.torso_head(rep[:, :, [0, 1, 4, 7, 8, 9, 10, 11, 14], :]) # [B, F, 9, 3]
        limb_output = F.adaptive_avg_pool2d(rep_limb, (8, 3)) # self.torso_head(rep[:, :, [0, 1, 4, 7, 8, 9, 10, 11, 14], :]) # [B, F, 9, 3]

        batch_pelvis = torso_output[:, :, 0, :].unsqueeze(2)
        batch_r_hip = torso_output[:, :, 1, :].unsqueeze(2)
        batch_l_hip = torso_output[:, :, 2, :].unsqueeze(2)
        batch_torso = torso_output[:, :, 3, :].unsqueeze(2)
        batch_neck = torso_output[:, :, 4, :].unsqueeze(2)
        batch_nose = torso_output[:, :, 5, :].unsqueeze(2)
        batch_head = torso_output[:, :, 6, :].unsqueeze(2)
        batch_l_shoulder = torso_output[:, :, 7, :].unsqueeze(2)
        batch_r_shoulder = torso_output[:, :, 8, :].unsqueeze(2)

        batch_r_knee = limb_output[:, :, 0, :].unsqueeze(2)
        batch_r_ankle = limb_output[:, :, 1, :].unsqueeze(2)
        batch_l_knee = limb_output[:, :, 2, :].unsqueeze(2)
        batch_l_ankle = limb_output[:, :, 3, :].unsqueeze(2)
        batch_l_elbow = limb_output[:, :, 4, :].unsqueeze(2)
        batch_l_wrist = limb_output[:, :, 5, :].unsqueeze(2)
        batch_r_elbow = limb_output[:, :, 6, :].unsqueeze(2)
        batch_r_wrist = limb_output[:, :, 7, :].unsqueeze(2)
         
        output = torch.cat([batch_pelvis, 
                            batch_r_hip, batch_r_knee, batch_r_ankle,
                            batch_l_hip, batch_l_knee, batch_l_ankle,
                            batch_torso, batch_neck, batch_nose, batch_head,
                            batch_l_shoulder, batch_l_elbow, batch_l_wrist,
                            batch_r_shoulder, batch_r_elbow, batch_r_wrist], dim=2)
        
        return output

class DHDSTformer_limb(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        # appendage id
        self.right_arm_id = 0
        self.left_arm_id  = 1
        self.right_leg_id = 2
        self.left_leg_id  = 3
        
        self.dstformer_backbone = load_backbone(args)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        self.dstformer_backbone = self.dstformer_backbone
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        
        self.right_arm_head = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.right_arm_head_angle = nn.Linear(args.dim_rep, 4) # 4: yaw1, pitch1, yaw2, pitch2
        self.right_arm_head_length = nn.Linear(args.dim_rep, 2) # 4: length1, length2
        self.left_arm_head  = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.left_arm_head_angle = nn.Linear(args.dim_rep, 4) # 4: yaw1, pitch1, yaw2, pitch2
        self.left_arm_head_length = nn.Linear(args.dim_rep, 2) # 4: length1, length2
        self.right_leg_head = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.right_leg_head_angle = nn.Linear(args.dim_rep, 4) # 4: yaw1, pitch1, yaw2, pitch2
        self.right_leg_head_length = nn.Linear(args.dim_rep, 2) # 4: length1, length2
        self.left_leg_head  = nn.Linear(args.dim_rep*3, args.dim_rep)
        self.left_leg_head_angle = nn.Linear(args.dim_rep, 4) # 4: yaw1, pitch1, yaw2, pitch2
        self.left_leg_head_length = nn.Linear(args.dim_rep, 2) # 4: length1, length2
        
    def forward(self, batch_input, batch_torso):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation
        
        # update batch size
        self.batch_size = batch_input.shape[0]
        self.batch_dh_model = BatchDHModel(None, batch_size=self.batch_size, num_frames=self.num_frames, data_type=self.data_type, device=batch_input.device)
        self.batch_dh_model.batch_size = self.batch_size

        # inference
        rep = self.dstformer_backbone.module.get_representation(batch_input)
        right_arm_rep = self.right_arm_head(rep[:, :, [14, 15, 16], :].flatten(2))
        left_arm_rep  =  self.left_arm_head(rep[:, :, [11, 12, 13], :].flatten(2))
        right_leg_rep = self.right_leg_head(rep[:, :, [1, 2, 3], :].flatten(2))
        left_leg_rep  =  self.left_leg_head(rep[:, :, [4, 5, 6], :].flatten(2))

        right_arm_angle  = self.right_arm_head_angle(right_arm_rep) # (N, F, 4)
        left_arm_angle   = self.left_arm_head_angle(left_arm_rep) # (N, F, 4)
        right_leg_angle  = self.right_leg_head_angle(right_leg_rep) # (N, F, 4)
        left_leg_angle   = self.left_leg_head_angle(left_leg_rep) # (N, F, 4)
        right_arm_length = self.right_arm_head_length(right_arm_rep) # (N, F, 2)
        left_arm_length  = self.left_arm_head_length(left_arm_rep) # (N, F, 2)
        right_leg_length = self.right_leg_head_length(right_leg_rep) # (N, F, 2)
        left_leg_length  = self.left_leg_head_length(left_leg_rep) # (N, F, 2)
        
        right_arm_output = torch.cat([right_arm_angle, right_arm_length], dim=-1)
        left_arm_output  = torch.cat([left_arm_angle, left_arm_length], dim=-1)
        right_leg_output = torch.cat([right_leg_angle, right_leg_length], dim=-1)
        left_leg_output  = torch.cat([left_leg_angle, left_leg_length], dim=-1)
        
        # update dh model
        self.batch_dh_model.set_batch_dh_model_from_dhdst_output(batch_torso, right_arm_output, left_arm_output, right_leg_output, left_leg_output)
        
        return self.batch_dh_model.get_batch_pose_3d()

class DHDSTformer_limb2(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        self.dstformer_backbone_limb = load_backbone(args)
        self.dstformer_backbone_limb = nn.DataParallel(self.dstformer_backbone_limb)
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone_limb.load_state_dict(checkpoint['model_pos'], strict=True)

    def forward(self, batch_input, batch_gt_torso):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation

        # inference
        rep_limb = self.dstformer_backbone_limb.module.get_representation(batch_input)
        limb_output = F.adaptive_avg_pool2d(rep_limb, (8, 3)) 
        batch_pelvis     = batch_gt_torso[:, :, 0, :].unsqueeze(2)
        batch_r_hip      = batch_gt_torso[:, :, 1, :].unsqueeze(2)
        batch_l_hip      = batch_gt_torso[:, :, 2, :].unsqueeze(2)
        batch_torso      = batch_gt_torso[:, :, 3, :].unsqueeze(2)
        batch_neck       = batch_gt_torso[:, :, 4, :].unsqueeze(2)
        batch_nose       = batch_gt_torso[:, :, 5, :].unsqueeze(2)
        batch_head       = batch_gt_torso[:, :, 6, :].unsqueeze(2)
        batch_l_shoulder = batch_gt_torso[:, :, 7, :].unsqueeze(2)
        batch_r_shoulder = batch_gt_torso[:, :, 8, :].unsqueeze(2)

        batch_r_knee  = limb_output[:, :, 0, :].unsqueeze(2)
        batch_r_ankle = limb_output[:, :, 1, :].unsqueeze(2)
        batch_l_knee  = limb_output[:, :, 2, :].unsqueeze(2)
        batch_l_ankle = limb_output[:, :, 3, :].unsqueeze(2)
        batch_l_elbow = limb_output[:, :, 4, :].unsqueeze(2)
        batch_l_wrist = limb_output[:, :, 5, :].unsqueeze(2)
        batch_r_elbow = limb_output[:, :, 6, :].unsqueeze(2)
        batch_r_wrist = limb_output[:, :, 7, :].unsqueeze(2)
         
        output = torch.cat([batch_pelvis, 
                            batch_r_hip, batch_r_knee, batch_r_ankle,
                            batch_l_hip, batch_l_knee, batch_l_ankle,
                            batch_torso, batch_neck, batch_nose, batch_head,
                            batch_l_shoulder, batch_l_elbow, batch_l_wrist,
                            batch_r_shoulder, batch_r_elbow, batch_r_wrist], dim=2)
        
        return output


class DHDSTformer_limb3(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        self.dstformer_backbone = load_backbone(args)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=True)

        # # freeze the backbone
        # for i, (name, param) in enumerate(self.dstformer_backbone_limb.named_parameters()):
        #     param.requires_grad = False


    def forward(self, batch_input, batch_gt_torso):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation
        
        # inference
        rep = self.dstformer_backbone.module.get_representation(batch_input)
        right_arm_output = F.adaptive_avg_pool2d(rep[:, :, [14, 15, 16], :], (2, 3)) 
        left_arm_output  = F.adaptive_avg_pool2d(rep[:, :, [11, 12, 13], :], (2, 3))
        right_leg_output = F.adaptive_avg_pool2d(rep[:, :, [1, 2, 3], :], (2, 3))
        left_leg_output  = F.adaptive_avg_pool2d(rep[:, :, [4, 5, 6], :], (2, 3))

        batch_pelvis     = batch_gt_torso[:, :, 0, :].unsqueeze(2)
        batch_r_hip      = batch_gt_torso[:, :, 1, :].unsqueeze(2)
        batch_l_hip      = batch_gt_torso[:, :, 2, :].unsqueeze(2)
        batch_torso      = batch_gt_torso[:, :, 3, :].unsqueeze(2)
        batch_neck       = batch_gt_torso[:, :, 4, :].unsqueeze(2)
        batch_nose       = batch_gt_torso[:, :, 5, :].unsqueeze(2)
        batch_head       = batch_gt_torso[:, :, 6, :].unsqueeze(2)
        batch_l_shoulder = batch_gt_torso[:, :, 7, :].unsqueeze(2)
        batch_r_shoulder = batch_gt_torso[:, :, 8, :].unsqueeze(2)

        batch_r_elbow = right_arm_output[:, :, 0, :].unsqueeze(2)
        batch_r_wrist = right_arm_output[:, :, 1, :].unsqueeze(2)
        batch_l_elbow = left_arm_output[:, :, 0, :].unsqueeze(2)
        batch_l_wrist = left_arm_output[:, :, 1, :].unsqueeze(2)
        batch_r_knee  = right_leg_output[:, :, 0, :].unsqueeze(2)
        batch_r_ankle = right_leg_output[:, :, 1, :].unsqueeze(2)
        batch_l_knee  = left_leg_output[:, :, 0, :].unsqueeze(2)
        batch_l_ankle = left_leg_output[:, :, 1, :].unsqueeze(2)
         
        output = torch.cat([batch_pelvis, 
                            batch_r_hip, batch_r_knee, batch_r_ankle,
                            batch_l_hip, batch_l_knee, batch_l_ankle,
                            batch_torso, batch_neck, batch_nose, batch_head,
                            batch_l_shoulder, batch_l_elbow, batch_l_wrist,
                            batch_r_shoulder, batch_r_elbow, batch_r_wrist], dim=2)
        
        return output
    
class DHDSTformer_limb4(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        # appendage id
        self.right_arm_id = 0
        self.left_arm_id  = 1
        self.right_leg_id = 2
        self.left_leg_id  = 3
        
        self.dstformer_backbone = load_backbone(args)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        self.dstformer_backbone = self.dstformer_backbone
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=True)

        self.r_knee_head  = nn.Linear(args.dim_rep, 3)
        self.r_ankle_head = nn.Linear(args.dim_rep, 3)
        self.l_knee_head  = nn.Linear(args.dim_rep, 3)
        self.l_ankle_head = nn.Linear(args.dim_rep, 3)
        self.r_elbow_head = nn.Linear(args.dim_rep, 3)
        self.r_wrist_head = nn.Linear(args.dim_rep, 3)
        self.l_elbow_head = nn.Linear(args.dim_rep, 3)
        self.l_wrist_head = nn.Linear(args.dim_rep, 3)
        
    def forward(self, batch_input, batch_gt_torso):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation

        # inference
        rep = self.dstformer_backbone.module.get_representation(batch_input)

        batch_pelvis     = batch_gt_torso[:, :, 0, :].unsqueeze(2)
        batch_r_hip      = batch_gt_torso[:, :, 1, :].unsqueeze(2)
        batch_l_hip      = batch_gt_torso[:, :, 2, :].unsqueeze(2)
        batch_torso      = batch_gt_torso[:, :, 3, :].unsqueeze(2)
        batch_neck       = batch_gt_torso[:, :, 4, :].unsqueeze(2)
        batch_nose       = batch_gt_torso[:, :, 5, :].unsqueeze(2)
        batch_head       = batch_gt_torso[:, :, 6, :].unsqueeze(2)
        batch_l_shoulder = batch_gt_torso[:, :, 7, :].unsqueeze(2)
        batch_r_shoulder = batch_gt_torso[:, :, 8, :].unsqueeze(2)

        batch_r_elbow = self.r_elbow_head(rep[:, :, 15, :]).unsqueeze(2)
        batch_r_wrist = self.r_wrist_head(rep[:, :, 16, :]).unsqueeze(2)
        batch_l_elbow = self.l_elbow_head(rep[:, :, 12, :]).unsqueeze(2)
        batch_l_wrist = self.l_wrist_head(rep[:, :, 13, :]).unsqueeze(2)
        batch_r_knee  = self.r_knee_head(rep[:, :, 2, :]).unsqueeze(2)
        batch_r_ankle = self.r_ankle_head(rep[:, :, 3, :]).unsqueeze(2)
        batch_l_knee  = self.l_knee_head(rep[:, :, 5, :]).unsqueeze(2)
        batch_l_ankle = self.l_ankle_head(rep[:, :, 6, :]).unsqueeze(2)
         
        output = torch.cat([batch_pelvis, 
                            batch_r_hip, batch_r_knee, batch_r_ankle, 
                            batch_l_hip, batch_l_knee, batch_l_ankle, 
                            batch_torso, batch_neck, batch_nose, batch_head, 
                            batch_l_shoulder, batch_l_elbow, batch_l_wrist, 
                            batch_r_shoulder, batch_r_elbow, batch_r_wrist], dim=2)
        
        return output
    
class DHDSTformer_limb5(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        # appendage id
        self.right_arm_id = 0
        self.left_arm_id  = 1
        self.right_leg_id = 2
        self.left_leg_id  = 3
        
        self.dstformer_backbone = load_backbone(args)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        self.dstformer_backbone = self.dstformer_backbone
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=True)

        # freeze the backbone
        for i, (name, param) in enumerate(self.dstformer_backbone.named_parameters()):
            param.requires_grad = False

        self.r_knee_head  = nn.Linear(args.dim_rep, 3)
        self.r_ankle_head = nn.Linear(args.dim_rep, 3)
        self.l_knee_head  = nn.Linear(args.dim_rep, 3)
        self.l_ankle_head = nn.Linear(args.dim_rep, 3)
        self.r_elbow_head = nn.Linear(args.dim_rep, 3)
        self.r_wrist_head = nn.Linear(args.dim_rep, 3)
        self.l_elbow_head = nn.Linear(args.dim_rep, 3)
        self.l_wrist_head = nn.Linear(args.dim_rep, 3)
        
    def forward(self, batch_input, batch_gt_torso):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation

        # inference
        rep = self.dstformer_backbone.module.get_representation(batch_input)

        batch_pelvis     = batch_gt_torso[:, :, 0, :].unsqueeze(2)
        batch_r_hip      = batch_gt_torso[:, :, 1, :].unsqueeze(2)
        batch_l_hip      = batch_gt_torso[:, :, 2, :].unsqueeze(2)
        batch_torso      = batch_gt_torso[:, :, 3, :].unsqueeze(2)
        batch_neck       = batch_gt_torso[:, :, 4, :].unsqueeze(2)
        batch_nose       = batch_gt_torso[:, :, 5, :].unsqueeze(2)
        batch_head       = batch_gt_torso[:, :, 6, :].unsqueeze(2)
        batch_l_shoulder = batch_gt_torso[:, :, 7, :].unsqueeze(2)
        batch_r_shoulder = batch_gt_torso[:, :, 8, :].unsqueeze(2)

        batch_r_elbow = self.r_elbow_head(rep[:, :, 15, :]).unsqueeze(2)
        batch_r_wrist = self.r_wrist_head(rep[:, :, 16, :]).unsqueeze(2)
        batch_l_elbow = self.l_elbow_head(rep[:, :, 12, :]).unsqueeze(2)
        batch_l_wrist = self.l_wrist_head(rep[:, :, 13, :]).unsqueeze(2)
        batch_r_knee  = self.r_knee_head(rep[:, :, 2, :]).unsqueeze(2)
        batch_r_ankle = self.r_ankle_head(rep[:, :, 3, :]).unsqueeze(2)
        batch_l_knee  = self.l_knee_head(rep[:, :, 5, :]).unsqueeze(2)
        batch_l_ankle = self.l_ankle_head(rep[:, :, 6, :]).unsqueeze(2)
         
        output = torch.cat([batch_pelvis, 
                            batch_r_hip, batch_r_knee, batch_r_ankle,
                            batch_l_hip, batch_l_knee, batch_l_ankle,
                            batch_torso, batch_neck, batch_nose, batch_head,
                            batch_l_shoulder, batch_l_elbow, batch_l_wrist,
                            batch_r_shoulder, batch_r_elbow, batch_r_wrist], dim=2)
        
        return output
    
class DHDSTformer_limb_all_in_one(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        # appendage id
        self.right_arm_id = 0
        self.left_arm_id  = 1
        self.right_leg_id = 2
        self.left_leg_id  = 3
        
        self.dstformer_backbone = load_backbone(args)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        self.dstformer_backbone = self.dstformer_backbone
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=True)

        self.limb_head  = nn.Linear(args.dim_rep, 3)
        
    def forward(self, batch_input, batch_gt_torso):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation

        # inference
        rep = self.dstformer_backbone.module.get_representation(batch_input)
        
        # batch_pelvis     = batch_gt_torso[:, :, 0, :].unsqueeze(2)
        # batch_r_hip      = batch_gt_torso[:, :, 1, :].unsqueeze(2)
        # batch_l_hip      = batch_gt_torso[:, :, 2, :].unsqueeze(2)
        # batch_torso      = batch_gt_torso[:, :, 3, :].unsqueeze(2)
        # batch_neck       = batch_gt_torso[:, :, 4, :].unsqueeze(2)
        # batch_nose       = batch_gt_torso[:, :, 5, :].unsqueeze(2)
        # batch_head       = batch_gt_torso[:, :, 6, :].unsqueeze(2)
        # batch_l_shoulder = batch_gt_torso[:, :, 7, :].unsqueeze(2)
        # batch_r_shoulder = batch_gt_torso[:, :, 8, :].unsqueeze(2)

        # batch_r_elbow = self.r_elbow_head(rep[:, :, 15, :]).unsqueeze(2)
        # batch_r_wrist = self.r_wrist_head(rep[:, :, 16, :]).unsqueeze(2)
        # batch_l_elbow = self.l_elbow_head(rep[:, :, 12, :]).unsqueeze(2)
        # batch_l_wrist = self.l_wrist_head(rep[:, :, 13, :]).unsqueeze(2)
        # batch_r_knee  = self.r_knee_head(rep[:, :, 2, :]).unsqueeze(2)
        # batch_r_ankle = self.r_ankle_head(rep[:, :, 3, :]).unsqueeze(2)
        # batch_l_knee  = self.l_knee_head(rep[:, :, 5, :]).unsqueeze(2)
        # batch_l_ankle = self.l_ankle_head(rep[:, :, 6, :]).unsqueeze(2)
         
        # output = torch.cat([batch_pelvis, 
        #                     batch_r_hip, batch_r_knee, batch_r_ankle,
        #                     batch_l_hip, batch_l_knee, batch_l_ankle,
        #                     batch_torso, batch_neck, batch_nose, batch_head,
        #                     batch_l_shoulder, batch_l_elbow, batch_l_wrist,
        #                     batch_r_shoulder, batch_r_elbow, batch_r_wrist], dim=2)
        
        raise NotImplementedError("Not implemented yet")
        
        return output
    
    
class DHDSTformer_right_arm(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        self.dstformer_backbone = load_backbone(args)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        self.dstformer_backbone = self.dstformer_backbone
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=True)

        self.head  = nn.Linear(args.dim_rep, 3)
        
    def forward(self, batch_input):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation

        # inference
        rep = self.dstformer_backbone.module.get_representation(batch_input)
        rep_right_arm = rep[:, :, [14, 15, 16], :] # (B, T, 3, dim_rep)
        output = self.head(rep_right_arm) # (B, T, 3, 3)
        
        return output
    
class DHDSTformer_right_arm2(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        self.dstformer_backbone = load_backbone(args)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        self.dstformer_backbone = self.dstformer_backbone
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=True)

        self.head  = nn.Linear(args.dim_rep, 3)
        
    def forward(self, batch_input):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation

        # inference
        rep = self.dstformer_backbone.module.get_representation(batch_input)
        rep_right_arm = rep[:, :, [0, 14, 15, 16], :] # (B, T, 4, dim_rep)
        output = self.head(rep_right_arm) # (B, T, 4, 3)
        
        return output
    
class DHDSTformer_right_arm3(nn.Module):
    def __init__(self, args, chk_filename='', dim_out=3, data_type=torch.float32):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_frames = args.clip_len
        self.data_type = data_type
        
        self.dstformer_backbone = load_backbone(args)
        self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
        self.dstformer_backbone = self.dstformer_backbone
        if chk_filename:
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        
    def forward(self, batch_input):
        # batch_x: (B, F, 17, 2) 2d pose
        # rep: (B, F, 17, dim_rep) hidden representation

        # inference
        rep = self.dstformer_backbone.module.get_representation(batch_input)
        output = F.adaptive_avg_pool2d(rep, (4, 3)) # (B, T, 4, 3)
        
        return output

# class DHDSTformer(nn.Module):
#     def __init__(self, chk_filename, args, dim_out=3, data_type=torch.float64):
#         super().__init__()
#         self.batch_size = args.batch_size
#         self.num_frames = args.clip_len
#         self.data_type = data_type
#         self.pred_length = args.pred_length
        
#         # appendage id
#         self.right_arm_id = 0
#         self.left_arm_id  = 1
#         self.right_leg_id = 2
#         self.left_leg_id  = 3
        
#         self.dstformer_backbone = load_backbone(args)
#         self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
#         self.dstformer_backbone = self.dstformer_backbone
#         checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
#         self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        
#         self.torso_head = nn.Linear(args.dim_rep, dim_out)
#         self.right_arm_head1 = nn.Linear(args.dim_rep*3, args.dim_rep)
#         self.right_arm_head2 = nn.Linear(args.dim_rep, 6) # 4: yaw1, pitch1, yaw2, pitch2, length1, length2
#         self.left_arm_head1  = nn.Linear(args.dim_rep*3, args.dim_rep)
#         self.left_arm_head2  = nn.Linear(args.dim_rep, 6) # 4: yaw1, pitch1, yaw2, pitch2, length1, length2
#         self.right_leg_head1 = nn.Linear(args.dim_rep*3, args.dim_rep)
#         self.right_leg_head2 = nn.Linear(args.dim_rep, 6) # 4: yaw1, pitch1, yaw2, pitch2, length1, length2
#         self.left_leg_head1  = nn.Linear(args.dim_rep*3, args.dim_rep)
#         self.left_leg_head2  = nn.Linear(args.dim_rep, 6) # 4: yaw1, pitch1, yaw2, pitch2, length1, length2
        
#         self.batch_zero = torch.zeros(self.batch_size, self.num_frames, dtype=self.data_type).cuda()
        
#     def batch_DH_matrix(self, batch_theta, batch_alpha, batch_a, batch_d):
#         m11 = torch.cos(batch_theta).unsqueeze(-1)
#         m12 = (-torch.sin(batch_theta)*torch.cos(batch_alpha)).unsqueeze(-1)
#         m13 = (torch.sin(batch_theta)*torch.sin(batch_alpha)).unsqueeze(-1)
#         m14 = (batch_a*torch.cos(batch_theta)).unsqueeze(-1)
#         m21 = torch.sin(batch_theta).unsqueeze(-1)
#         m22 = (torch.cos(batch_theta)*torch.cos(batch_alpha)).unsqueeze(-1)
#         m23 = (-torch.cos(batch_theta)*torch.sin(batch_alpha)).unsqueeze(-1)
#         m24 = (batch_a*torch.sin(batch_theta)).unsqueeze(-1)
#         m31 = torch.zeros_like(batch_theta).unsqueeze(-1)
#         m32 = torch.sin(batch_alpha).unsqueeze(-1)
#         m33 = torch.cos(batch_alpha).unsqueeze(-1)
#         m34 = batch_d.unsqueeze(-1)
#         m41 = torch.zeros_like(batch_theta).unsqueeze(-1)
#         m42 = torch.zeros_like(batch_theta).unsqueeze(-1)
#         m43 = torch.zeros_like(batch_theta).unsqueeze(-1)
#         m44 = torch.ones_like(batch_theta).unsqueeze(-1)
#         row1 = torch.concat([m11, m12, m13, m14], dim=-1)
#         row2 = torch.concat([m21, m22, m23, m24], dim=-1)
#         row3 = torch.concat([m31, m32, m33, m34], dim=-1)
#         row4 = torch.concat([m41, m42, m43, m44], dim=-1)
#         return torch.stack([row1, row2, row3, row4], dim=-1).transpose(2, 3)

#     def forward(self, batch_input, batch_label=None, length_type='frame', ref_frame=0):
#         # batch_x: (B, F, 17, 2) 2d pose
#         # rep: (B, F, 17, dim_rep) hidden representation
#         # output: (B, F, 17, 49) - 0:26: torso, 27:30: right arm, 31:34: left arm, 35:38: right leg, 39:42: left leg
        
#         # update batch size
#         self.batch_size = batch_input.shape[0]
#         self.batch_zero = torch.zeros(self.batch_size, self.num_frames, dtype=self.data_type).cuda()
        
#         # inference
#         rep = self.dstformer_backbone.module.get_representation(batch_input)
#         torso_output = self.torso_head(rep[:, :, [0, 1, 4, 7, 8, 9, 10, 11, 14], :]) # (N, F, 9, 3)
#         right_arm_output = self.right_arm_head2(self.right_arm_head1(rep[:, :, [14, 15, 16], :].flatten(2))) # (N, F, 4)
#         left_arm_output  =  self.left_arm_head2(self.left_arm_head1(rep[:, :, [11, 12, 13], :].flatten(2))) # (N, F, 4)
#         right_leg_output = self.right_leg_head2(self.right_leg_head1(rep[:, :, [1, 2, 3], :].flatten(2))) # (N, F, 4)
#         left_leg_output  =  self.left_leg_head2(self.left_leg_head1(rep[:, :, [4, 5, 6], :].flatten(2))) # (N, F, 4)

#         # update dh model
#         # self.update_limb_lens(batch_label, length_type, ref_frame)
#         # self.update_batch_keypoints(self.torso_output)
#         # self.update_body_reference_frame()
#         # self.update_root_tf()  
#         # self.batch_r_elbow, self.batch_r_wrist = self.forward_appendage(0, self.right_arm_output[:, :, 0], self.right_arm_output[:, :, 1], self.right_arm_output[:, :, 2], self.right_arm_output[:, :, 3], self.batch_ra_l1_length, self.batch_ra_l2_length)
#         # self.batch_l_elbow, self.batch_l_wrist = self.forward_appendage(1, self.left_arm_output[:, :, 0], self.left_arm_output[:, :, 1], self.left_arm_output[:, :, 2], self.left_arm_output[:, :, 3], self.batch_la_l1_length, self.batch_la_l2_length)
#         # self.batch_r_knee, self.batch_r_ankle  = self.forward_appendage(2, self.right_leg_output[:, :, 0], self.right_leg_output[:, :, 1], self.right_leg_output[:, :, 2], self.right_leg_output[:, :, 3], self.batch_rl_l1_length, self.batch_rl_l2_length)
#         # self.batch_l_knee, self.batch_l_ankle  = self.forward_appendage(3, self.left_leg_output[:, :, 0], self.left_leg_output[:, :, 1], self.left_leg_output[:, :, 2], self.left_leg_output[:, :, 3], self.batch_ll_l1_length, self.batch_ll_l2_length)
#         # return self.get_pose_output()
#         # return torch.concat([torso_output.flatten(2, 3), self.right_arm_output, self.left_arm_output, self.right_leg_output, self.left_leg_output], dim=-1) # (B, F, 43) #
#         # return torso_output, right_arm_output, left_arm_output, right_leg_output, left_leg_output # tuple
        
#         if (batch_label != None) and (self.pred_length == False):
#             batch_ra_l1_length, batch_ra_l2_length, batch_la_l1_length, batch_la_l2_length, batch_rl_l1_length, batch_rl_l2_length, batch_ll_l1_length, batch_ll_l2_length = self.update_limb_lens(batch_label, length_type, ref_frame)
#         else:
#             batch_ra_l1_length, batch_ra_l2_length = right_arm_output[:, :, 4], right_arm_output[:, :, 5]
#             batch_la_l1_length, batch_la_l2_length =  left_arm_output[:, :, 4],  left_arm_output[:, :, 5] 
#             batch_rl_l1_length, batch_rl_l2_length = right_leg_output[:, :, 4], right_leg_output[:, :, 5]
#             batch_ll_l1_length, batch_ll_l2_length =  left_leg_output[:, :, 4],  left_leg_output[:, :, 5] 
            
#         batch_pelvis, batch_r_hip, batch_l_hip, batch_torso, batch_neck, batch_nose, batch_head, batch_l_shoulder, batch_r_shoulder = self.update_batch_keypoints(torso_output)
#         batch_body_R = self.update_body_reference_frame(batch_pelvis, batch_l_hip)
#         root_tf = self.update_root_tf(batch_body_R, batch_r_shoulder, batch_l_shoulder, batch_r_hip, batch_l_hip)  
#         batch_r_elbow, batch_r_wrist = self.forward_appendage(root_tf[:, :, self.right_arm_id, :, :], right_arm_output[:, :, 0], right_arm_output[:, :, 1], right_arm_output[:, :, 2], right_arm_output[:, :, 3], batch_ra_l1_length, batch_ra_l2_length)
#         batch_l_elbow, batch_l_wrist = self.forward_appendage(root_tf[:, :, self.left_arm_id,  :, :], left_arm_output[:, :, 0],  left_arm_output[:, :, 1],  left_arm_output[:, :, 2],  left_arm_output[:, :, 3],  batch_la_l1_length, batch_la_l2_length)
#         batch_r_knee,  batch_r_ankle = self.forward_appendage(root_tf[:, :, self.right_leg_id, :, :], right_leg_output[:, :, 0], right_leg_output[:, :, 1], right_leg_output[:, :, 2], right_leg_output[:, :, 3], batch_rl_l1_length, batch_rl_l2_length)
#         batch_l_knee,  batch_l_ankle = self.forward_appendage(root_tf[:, :, self.left_leg_id,  :, :], left_leg_output[:, :, 0],  left_leg_output[:, :, 1],  left_leg_output[:, :, 2],  left_leg_output[:, :, 3],  batch_ll_l1_length, batch_ll_l2_length)
        
#         pose_3d = torch.zeros(self.batch_size, self.num_frames, 17, 3, dtype=self.data_type).cuda() # leaf node
#         pose_3d[:, :, 0]  = batch_pelvis 
#         pose_3d[:, :, 1]  = batch_r_hip
#         pose_3d[:, :, 2]  = batch_r_knee 
#         pose_3d[:, :, 3]  = batch_r_ankle
#         pose_3d[:, :, 4]  = batch_l_hip
#         pose_3d[:, :, 5]  = batch_l_knee 
#         pose_3d[:, :, 6]  = batch_l_ankle
#         pose_3d[:, :, 7]  = batch_torso
#         pose_3d[:, :, 8]  = batch_neck
#         pose_3d[:, :, 9]  = batch_nose
#         pose_3d[:, :, 10] = batch_head
#         pose_3d[:, :, 11] = batch_l_shoulder 
#         pose_3d[:, :, 12] = batch_l_elbow
#         pose_3d[:, :, 13] = batch_l_wrist
#         pose_3d[:, :, 14] = batch_r_shoulder
#         pose_3d[:, :, 15] = batch_r_elbow 
#         pose_3d[:, :, 16] = batch_r_wrist 
        
#         return pose_3d
        
#     def dt_test_forward(self, batch_torso_output, batch_dh_angles, batch_pose, length_type='mean', ref_frame=0):    
#         batch_ra_l1_length, batch_ra_l2_length, batch_la_l1_length, batch_la_l2_length, batch_rl_l1_length, batch_rl_l2_length, batch_ll_l1_length, batch_ll_l2_length = self.update_limb_lens(batch_pose, length_type, ref_frame)
#         batch_pelvis, batch_r_hip, batch_l_hip, batch_torso, batch_neck, batch_nose, batch_head, batch_l_shoulder, batch_r_shoulder = self.update_batch_keypoints(batch_torso_output)
#         batch_body_R = self.update_body_reference_frame(batch_pelvis, batch_l_hip)
#         root_tf = self.update_root_tf(batch_body_R, batch_r_shoulder, batch_l_shoulder, batch_r_hip, batch_l_hip)  
#         right_arm_output = batch_dh_angles[:, :, [2, 3, 4, 5]] # (N, F, 4)
#         left_arm_output  = batch_dh_angles[:, :, [6, 7, 8, 9]]
#         right_leg_output = batch_dh_angles[:, :, [10, 11, 12, 13]]
#         left_leg_output  = batch_dh_angles[:, :, [14, 15, 16, 17]]
#         batch_r_elbow, batch_r_wrist = self.forward_appendage(root_tf[:, :, self.right_arm_id, :, :], right_arm_output[:, :, 0], right_arm_output[:, :, 1], right_arm_output[:, :, 2], right_arm_output[:, :, 3], batch_ra_l1_length, batch_ra_l2_length)
#         batch_l_elbow, batch_l_wrist = self.forward_appendage(root_tf[:, :, self.left_arm_id,  :, :], left_arm_output[:, :, 0],  left_arm_output[:, :, 1],  left_arm_output[:, :, 2],  left_arm_output[:, :, 3],  batch_la_l1_length, batch_la_l2_length)
#         batch_r_knee,  batch_r_ankle = self.forward_appendage(root_tf[:, :, self.right_leg_id, :, :], right_leg_output[:, :, 0], right_leg_output[:, :, 1], right_leg_output[:, :, 2], right_leg_output[:, :, 3], batch_rl_l1_length, batch_rl_l2_length)
#         batch_l_knee,  batch_l_ankle = self.forward_appendage(root_tf[:, :, self.left_leg_id,  :, :], left_leg_output[:, :, 0],  left_leg_output[:, :, 1],  left_leg_output[:, :, 2],  left_leg_output[:, :, 3],  batch_ll_l1_length, batch_ll_l2_length)
        
#         pose_3d = torch.zeros(self.batch_size, self.num_frames, 17, 3, dtype=self.data_type).cuda() # leaf node
#         pose_3d[:, :, 0]  = batch_pelvis 
#         pose_3d[:, :, 1]  = batch_r_hip
#         pose_3d[:, :, 2]  = batch_r_knee 
#         pose_3d[:, :, 3]  = batch_r_ankle
#         pose_3d[:, :, 4]  = batch_l_hip
#         pose_3d[:, :, 5]  = batch_l_knee 
#         pose_3d[:, :, 6]  = batch_l_ankle
#         pose_3d[:, :, 7]  = batch_torso
#         pose_3d[:, :, 8]  = batch_neck
#         pose_3d[:, :, 9]  = batch_nose
#         pose_3d[:, :, 10] = batch_head
#         pose_3d[:, :, 11] = batch_l_shoulder 
#         pose_3d[:, :, 12] = batch_l_elbow
#         pose_3d[:, :, 13] = batch_l_wrist
#         pose_3d[:, :, 14] = batch_r_shoulder
#         pose_3d[:, :, 15] = batch_r_elbow 
#         pose_3d[:, :, 16] = batch_r_wrist 
        
#         return pose_3d
    
#     def update_limb_lens(self, x, length_type='frame', ref_frame=0):
#         '''
#             Input: (N, T, 17, 3)
#             Output: (N, T, 16)
#         '''
#         limbs_id = [[0,1], [1,2], [2,3],
#             [0,4], [4,5], [5,6],
#             [0,7], [7,8], [8,9], [9,10],
#             [8,11], [11,12], [12,13],
#             [8,14], [14,15], [15,16]
#             ]
#         limbs = x[:,:,limbs_id,:]
#         limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
#         batch_limb_lens = torch.norm(limbs, dim=-1) # [B, F, 16]
        
#         if length_type == 'each':
#             batch_ra_l1_length = batch_limb_lens[:, :, 14] # [B, F]
#             batch_ra_l2_length = batch_limb_lens[:, :, 15] # [B, F]
#             batch_la_l1_length = batch_limb_lens[:, :, 11] # [B, F]
#             batch_la_l2_length = batch_limb_lens[:, :, 12] # [B, F]
#             batch_rl_l1_length = batch_limb_lens[:, :, 1] # [B, F]
#             batch_rl_l2_length = batch_limb_lens[:, :, 2] # [B, F]
#             batch_ll_l1_length = batch_limb_lens[:, :, 4] # [B, F]
#             batch_ll_l2_length = batch_limb_lens[:, :, 5] # [B, F]
#         elif length_type == 'mean':
#             batch_init_limb_lens = batch_limb_lens.mean(dim=1)
#             batch_ra_l1_length = batch_init_limb_lens[:, 14].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
#             batch_ra_l2_length = batch_init_limb_lens[:, 15].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
#             batch_la_l1_length = batch_init_limb_lens[:, 11].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
#             batch_la_l2_length = batch_init_limb_lens[:, 12].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
#             batch_rl_l1_length = batch_init_limb_lens[:, 1].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
#             batch_rl_l2_length = batch_init_limb_lens[:, 2].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
#             batch_ll_l1_length = batch_init_limb_lens[:, 4].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
#             batch_ll_l2_length = batch_init_limb_lens[:, 5].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
#         elif length_type == 'frame':
#             batch_init_limb_lens = batch_limb_lens[:, ref_frame, :] # [B, 16]
#             batch_ra_l1_length = batch_init_limb_lens[:, 14].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
#             batch_ra_l2_length = batch_init_limb_lens[:, 15].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
#             batch_la_l1_length = batch_init_limb_lens[:, 11].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
#             batch_la_l2_length = batch_init_limb_lens[:, 12].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
#             batch_rl_l1_length = batch_init_limb_lens[:, 1].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
#             batch_rl_l2_length = batch_init_limb_lens[:, 2].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
#             batch_ll_l1_length = batch_init_limb_lens[:, 4].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
#             batch_ll_l2_length = batch_init_limb_lens[:, 5].unsqueeze(1).repeat(1, self.num_frames) # [B, F]
            
#         return batch_ra_l1_length, batch_ra_l2_length, batch_la_l1_length, batch_la_l2_length, batch_rl_l1_length, batch_rl_l2_length, batch_ll_l1_length, batch_ll_l2_length

#     def update_batch_keypoints(self, torso_output):
#         '''
#             Input: (N, T, 9, 3)
#         '''
#         batch_pelvis     = torso_output[:, :, 0] # [B, F, 3]
#         batch_r_hip      = torso_output[:, :, 1] # [B, F, 3]
#         batch_l_hip      = torso_output[:, :, 2] # [B, F, 3]
#         batch_torso      = torso_output[:, :, 3] # [B, F, 3]
#         batch_neck       = torso_output[:, :, 4] # [B, F, 3]
#         batch_nose       = torso_output[:, :, 5] # [B, F, 3]
#         batch_head       = torso_output[:, :, 6] # [B, F, 3]
#         batch_l_shoulder = torso_output[:, :, 7] # [B, F, 3]
#         batch_r_shoulder = torso_output[:, :, 8] # [B, F, 3]
        
#         return batch_pelvis, batch_r_hip, batch_l_hip, batch_torso, batch_neck, batch_nose, batch_head, batch_l_shoulder, batch_r_shoulder
        
#     def update_body_reference_frame(self, batch_pelvis, batch_l_hip):
#         ## get body reference frame
#         # z axis
#         batch_z_axis = torch.tensor([0, 0, 1], dtype=self.data_type).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.num_frames, 1).cuda() # [B, F, 3]
#         # y axis
#         batch_y_axis = batch_l_hip - batch_pelvis # [B, F, 3]
#         batch_y_axis = batch_y_axis.type(self.data_type)
#         batch_y_axis[:, :, 2] = 0
#         batch_y_axis_mag = torch.norm(batch_y_axis, dim=2).unsqueeze(-1)
#         batch_y_axis = batch_y_axis/batch_y_axis_mag # '/=' is inplace operation
#         # x axis
#         batch_x_axis = torch.cross(batch_y_axis, batch_z_axis, dim=2) # [B, F, 3]
#         # body_R
#         batch_body_R = torch.cat([batch_x_axis.unsqueeze(-1), batch_y_axis.unsqueeze(-1), batch_z_axis.unsqueeze(-1)], dim=-1).transpose(2, 3) # [B, F, 3, 3]
        
#         return batch_body_R
        
#     def update_root_tf(self, batch_body_R, batch_r_shoulder, batch_l_shoulder, batch_r_hip, batch_l_hip):
#         rot_y_180 = torch.tensor(Rotation.from_rotvec(np.array([0, np.pi, 0])).as_matrix(), dtype=self.data_type).cuda() # [3, 3]
#         root_tf = torch.eye(4, dtype=self.data_type).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.num_frames, 4, 1, 1).cuda()
#         root_tf[:, :, :, :3, :3] = torch.matmul(rot_y_180.clone(), batch_body_R.clone().to(rot_y_180.device)).unsqueeze(2).repeat(1, 1, 4, 1, 1) # [B, F, 4, 3, 3]
#         root_tf[:, :, self.right_arm_id, :3, 3] = batch_r_shoulder # right arm
#         root_tf[:, :, self.left_arm_id , :3, 3] = batch_l_shoulder # left arm
#         root_tf[:, :, self.right_leg_id, :3, 3] = batch_r_hip # right leg
#         root_tf[:, :, self.left_leg_id , :3, 3] = batch_l_hip # left leg
        
#         return root_tf
        
#     def forward_appendage(self, root_tf, batch_link1_azim, batch_link1_elev, batch_link2_azim, batch_link2_elev, batch_l1_length, batch_l2_length):
#         batch_link1_mat = self.batch_DH_matrix(batch_link1_azim, batch_link1_elev, self.batch_zero, self.batch_zero) # [B, F, 4, 4]
#         batch_link2_mat = self.batch_DH_matrix(batch_link2_azim, batch_link2_elev, self.batch_zero, batch_l1_length) # [B, F, 4, 4]
#         batch_terminal_mat = self.batch_DH_matrix(self.batch_zero, self.batch_zero, self.batch_zero, batch_l2_length) # [B, F, 4, 4]
        
#         batch_link1_tf = torch.matmul(root_tf, batch_link1_mat) # [B, F, 4, 4]
#         batch_link2_tf = torch.matmul(batch_link1_tf, batch_link2_mat) # [B, F, 4, 4]
#         batch_terminal_tf = torch.matmul(batch_link2_tf, batch_terminal_mat) # [B, F, 4, 4]
#         batch_link2_origin = batch_link2_tf[:, :, :3, 3] # [B, F, 3]
#         batch_terminal_origin = batch_terminal_tf[:, :, :3, 3] # [B, F, 3]
        
#         return batch_link2_origin, batch_terminal_origin
        
#     def get_pose_output(self, ):
#         pose_3d = torch.zeros(self.batch_size, self.num_frames, 17, 3, dtype=self.data_type).cuda() # leaf node
#         pose_3d[:, :, 0]  = self.batch_pelvis 
#         pose_3d[:, :, 1]  = self.batch_r_hip
#         pose_3d[:, :, 2]  = self.batch_r_knee 
#         pose_3d[:, :, 3]  = self.batch_r_ankle
#         pose_3d[:, :, 4]  = self.batch_l_hip
#         pose_3d[:, :, 5]  = self.batch_l_knee 
#         pose_3d[:, :, 6]  = self.batch_l_ankle
#         pose_3d[:, :, 7]  = self.batch_torso
#         pose_3d[:, :, 8]  = self.batch_neck
#         pose_3d[:, :, 9]  = self.batch_nose
#         pose_3d[:, :, 10] = self.batch_head
#         pose_3d[:, :, 11] = self.batch_l_shoulder 
#         pose_3d[:, :, 12] = self.batch_l_elbow
#         pose_3d[:, :, 13] = self.batch_l_wrist
#         pose_3d[:, :, 14] = self.batch_r_shoulder
#         pose_3d[:, :, 15] = self.batch_r_elbow 
#         pose_3d[:, :, 16] = self.batch_r_wrist 
#         return pose_3d
    
    
    
    

# ## 함수화 하기
# import sys
# sys.path.append('/home/hrai/codes/PoseAdaptor')
# from lib_import import *
# from my_utils import *

# from functools import partial
# from lib.model.DSTformer import DSTformer
# from lib.utils.learning import * # load_backbone

# class DHDSTformer(nn.Module):
#     def __init__(self, chk_filename, args, dim_out=3):
#         super().__init__()
#         self.batch_size = args.batch_size
#         self.num_frames = args.clip_len
        
#         # appendage id
#         self.right_arm_id = 0
#         self.left_arm_id  = 1
#         self.right_leg_id = 2
#         self.left_leg_id  = 3
        
#         self.dstformer_backbone = load_backbone(args)
#         self.dstformer_backbone = nn.DataParallel(self.dstformer_backbone)
#         self.dstformer_backbone = self.dstformer_backbone
#         checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
#         self.dstformer_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        
#         self.torso_head = nn.Linear(args.dim_rep, dim_out)
#         self.right_arm_head1 = nn.Linear(args.dim_rep*3, args.dim_rep)
#         self.right_arm_head2 = nn.Linear(args.dim_rep, 4)
#         self.left_arm_head1  = nn.Linear(args.dim_rep*3, args.dim_rep)
#         self.left_arm_head2  = nn.Linear(args.dim_rep, 4)
#         self.right_leg_head1 = nn.Linear(args.dim_rep*3, args.dim_rep)
#         self.right_leg_head2 = nn.Linear(args.dim_rep, 4)
#         self.left_leg_head1  = nn.Linear(args.dim_rep*3, args.dim_rep)
#         self.left_leg_head2  = nn.Linear(args.dim_rep, 4)
        
#         self.rot_y_180 = torch.tensor(Rotation.from_rotvec(np.array([0, np.pi, 0])).as_matrix(), dtype=torch.float32).cuda() # [3, 3]
#         self.batch_zero = torch.zeros(self.batch_size, self.num_frames, dtype=torch.float32).cuda()
        
#     def batch_DH_matrix(self, batch_theta, batch_alpha, batch_a, batch_d):
#         m11 = torch.cos(batch_theta)
#         m11 = m11.unsqueeze(-1)
#         m12 = (-torch.sin(batch_theta)*torch.cos(batch_alpha))
#         m12 = m12.unsqueeze(-1)
#         m13 = (torch.sin(batch_theta)*torch.sin(batch_alpha))
#         m13 = m13.unsqueeze(-1)
#         m14 = (batch_a*torch.cos(batch_theta))
#         m14 = m14.unsqueeze(-1)
#         m21 = torch.sin(batch_theta)
#         m21 = m21.unsqueeze(-1)
#         m22 = (torch.cos(batch_theta)*torch.cos(batch_alpha))
#         m22 = m22.unsqueeze(-1)
#         m23 = (-torch.cos(batch_theta)*torch.sin(batch_alpha))
#         m23 = m23.unsqueeze(-1)
#         m24 = (batch_a*torch.sin(batch_theta))
#         m24 = m24.unsqueeze(-1)
#         m31 = torch.zeros_like(batch_theta)
#         m31 = m31.unsqueeze(-1)
#         m32 = torch.sin(batch_alpha)
#         m32 = m32.unsqueeze(-1)
#         m33 = torch.cos(batch_alpha)
#         m33 = m33.unsqueeze(-1)
#         m34 = batch_d
#         m34 = m34.unsqueeze(-1)
#         m41 = torch.zeros_like(batch_theta)
#         m41 = m41.unsqueeze(-1)
#         m42 = torch.zeros_like(batch_theta)
#         m42 = m42.unsqueeze(-1)
#         m43 = torch.zeros_like(batch_theta)
#         m43 = m43.unsqueeze(-1)
#         m44 = torch.ones_like(batch_theta)
#         m44 = m44.unsqueeze(-1)
#         row1 = torch.concat([m11, m12, m13, m14], dim=-1)
#         row2 = torch.concat([m21, m22, m23, m24], dim=-1)
#         row3 = torch.concat([m31, m32, m33, m34], dim=-1)
#         row4 = torch.concat([m41, m42, m43, m44], dim=-1)
#         return torch.stack([row1, row2, row3, row4], dim=-1).transpose(2, 3)

#     def forward(self, batch_input, batch_label, length_type='frame', ref_frame=0):
#         # batch_x: (B, F, 17, 2) 2d pose
#         # rep: (B, F, 17, dim_rep) hidden representation
#         # output: (B, F, 17, 49) - 0:26: torso, 27:30: right arm, 31:34: left arm, 35:38: right leg, 39:42: left leg
        
#         # update batch size
#         self.batch_size = batch_input.shape[0]
#         self.batch_zero = torch.zeros(self.batch_size, self.num_frames, dtype=torch.float32).cuda()
        
#         # inference
#         rep = self.dstformer_backbone.module.get_representation(batch_input)
#         self.torso_output = self.torso_head(rep[:, :, [0, 1, 4, 7, 8, 9, 10, 11, 14], :]) # (N, F, 9, 3)
#         self.right_arm_output = self.right_arm_head2(self.right_arm_head1(rep[:, :, [14, 15, 16], :].flatten(2))) # (N, F, 4)
#         self.left_arm_output  =  self.left_arm_head2(self.left_arm_head1(rep[:, :, [11, 12, 13], :].flatten(2))) # (N, F, 4)
#         self.right_leg_output = self.right_leg_head2(self.right_leg_head1(rep[:, :, [1, 2, 3], :].flatten(2))) # (N, F, 4)
#         self.left_leg_output  =  self.left_leg_head2(self.left_leg_head1(rep[:, :, [4, 5, 6], :].flatten(2))) # (N, F, 4)
        
#         # update dh model
#         self.update_limb_lens(batch_label, length_type, ref_frame)
#         self.update_batch_keypoints(self.torso_output)
#         self.update_body_reference_frame()
#         self.update_root_tf()  
#         self.batch_r_elbow, self.batch_r_wrist = self.forward_appendage(0, self.right_arm_output[:, :, 0], self.right_arm_output[:, :, 1], self.right_arm_output[:, :, 2], self.right_arm_output[:, :, 3], self.batch_ra_l1_length, self.batch_ra_l2_length)
#         self.batch_l_elbow, self.batch_l_wrist = self.forward_appendage(1, self.left_arm_output[:, :, 0], self.left_arm_output[:, :, 1], self.left_arm_output[:, :, 2], self.left_arm_output[:, :, 3], self.batch_la_l1_length, self.batch_la_l2_length)
#         self.batch_r_knee, self.batch_r_ankle = self.forward_appendage(2, self.right_leg_output[:, :, 0], self.right_leg_output[:, :, 1], self.right_leg_output[:, :, 2], self.right_leg_output[:, :, 3], self.batch_rl_l1_length, self.batch_rl_l2_length)
#         self.batch_l_knee, self.batch_l_ankle = self.forward_appendage(3, self.left_leg_output[:, :, 0], self.left_leg_output[:, :, 1], self.left_leg_output[:, :, 2], self.left_leg_output[:, :, 3], self.batch_ll_l1_length, self.batch_ll_l2_length)
#         return self.get_pose_output()
#         # return torch.concat([torso_output.flatten(2, 3), self.right_arm_output, self.left_arm_output, self.right_leg_output, self.left_leg_output], dim=-1) # (B, F, 43) #
#         # return torso_output, right_arm_output, left_arm_output, right_leg_output, left_leg_output # tuple
        
#     def dt_test_forward(self, batch_torso_output, batch_dh_angles, batch_pose, length_type='frame', ref_frame=0):
#         self.update_limb_lens(batch_pose, length_type, ref_frame)
#         self.update_batch_keypoints(batch_torso_output)
#         self.update_body_reference_frame()
#         self.update_root_tf()  
#         self.right_arm_output = batch_dh_angles[:, :, [2, 3, 4, 5]] # (N, F, 4)
#         self.left_arm_output  = batch_dh_angles[:, :, [6, 7, 8, 9]]
#         self.right_leg_output = batch_dh_angles[:, :, [10, 11, 12, 13]]
#         self.left_leg_output  = batch_dh_angles[:, :, [14, 15, 16, 17]]
#         self.batch_r_elbow, self.batch_r_wrist = self.forward_appendage(self.right_arm_id, self.right_arm_output[:, :, 0], self.right_arm_output[:, :, 1], self.right_arm_output[:, :, 2], self.right_arm_output[:, :, 3], self.batch_ra_l1_length, self.batch_ra_l2_length)
#         self.batch_l_elbow, self.batch_l_wrist = self.forward_appendage(self.left_arm_id , self.left_arm_output[:, :, 0], self.left_arm_output[:, :, 1], self.left_arm_output[:, :, 2], self.left_arm_output[:, :, 3], self.batch_la_l1_length, self.batch_la_l2_length)
#         self.batch_r_knee, self.batch_r_ankle  = self.forward_appendage(self.right_leg_id, self.right_leg_output[:, :, 0], self.right_leg_output[:, :, 1], self.right_leg_output[:, :, 2], self.right_leg_output[:, :, 3], self.batch_rl_l1_length, self.batch_rl_l2_length)
#         self.batch_l_knee, self.batch_l_ankle  = self.forward_appendage(self.left_leg_id , self.left_leg_output[:, :, 0], self.left_leg_output[:, :, 1], self.left_leg_output[:, :, 2], self.left_leg_output[:, :, 3], self.batch_ll_l1_length, self.batch_ll_l2_length)
#         return self.get_pose_output()
    
#     def update_limb_lens(self, x, length_type='frame', ref_frame=0):
#         '''
#             Input: (N, T, 17, 3)
#             Output: (N, T, 16)
#         '''
#         limbs_id = [[0,1], [1,2], [2,3],
#             [0,4], [4,5], [5,6],
#             [0,7], [7,8], [8,9], [9,10],
#             [8,11], [11,12], [12,13],
#             [8,14], [14,15], [15,16]
#             ]
#         limbs = x[:,:,limbs_id,:]
#         limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
#         batch_limb_lens = torch.norm(limbs, dim=-1) # [B, F, 16]
        
#         if length_type == 'each':
#             self.batch_ra_l1_length = batch_limb_lens[:, :, 14] # [B, F]
#             self.batch_ra_l2_length = batch_limb_lens[:, :, 15] # [B, F]
#             self.batch_la_l1_length = batch_limb_lens[:, :, 11] # [B, F]
#             self.batch_la_l2_length = batch_limb_lens[:, :, 12] # [B, F]
#             self.batch_rl_l1_length = batch_limb_lens[:, :, 1] # [B, F]
#             self.batch_rl_l2_length = batch_limb_lens[:, :, 2] # [B, F]
#             self.batch_ll_l1_length = batch_limb_lens[:, :, 4] # [B, F]
#             self.batch_ll_l2_length = batch_limb_lens[:, :, 5] # [B, F]
#         elif length_type == 'mean':
#             batch_init_limb_lens = batch_limb_lens.mean(dim=1)
#             self.batch_ra_l1_length = batch_init_limb_lens[:, 14]
#             self.batch_ra_l1_length = self.batch_ra_l1_length.unsqueeze(1)
#             self.batch_ra_l1_length = self.batch_ra_l1_length.repeat(1, self.num_frames) # [B, F]
#             self.batch_ra_l2_length = batch_init_limb_lens[:, 15]
#             self.batch_ra_l2_length = self.batch_ra_l2_length.unsqueeze(1)
#             self.batch_ra_l2_length = self.batch_ra_l2_length.repeat(1, self.num_frames) # [B, F]
#             self.batch_la_l1_length = batch_init_limb_lens[:, 11]
#             self.batch_la_l1_length = self.batch_la_l1_length.unsqueeze(1)
#             self.batch_la_l1_length = self.batch_la_l1_length.repeat(1, self.num_frames) # [B, F]
#             self.batch_la_l2_length = batch_init_limb_lens[:, 12]
#             self.batch_la_l2_length = self.batch_la_l2_length.unsqueeze(1)
#             self.batch_la_l2_length = self.batch_la_l2_length.repeat(1, self.num_frames) # [B, F]
#             self.batch_rl_l1_length = batch_init_limb_lens[:, 1] 
#             self.batch_rl_l1_length = self.batch_rl_l1_length.unsqueeze(1)
#             self.batch_rl_l1_length = self.batch_rl_l1_length.repeat(1, self.num_frames) # [B, F]
#             self.batch_rl_l2_length = batch_init_limb_lens[:, 2] 
#             self.batch_rl_l2_length = self.batch_rl_l2_length.unsqueeze(1)
#             self.batch_rl_l2_length = self.batch_rl_l2_length.repeat(1, self.num_frames) # [B, F]
#             self.batch_ll_l1_length = batch_init_limb_lens[:, 4] 
#             self.batch_ll_l1_length = self.batch_ll_l1_length.unsqueeze(1)
#             self.batch_ll_l1_length = self.batch_ll_l1_length.repeat(1, self.num_frames) # [B, F]
#             self.batch_ll_l2_length = batch_init_limb_lens[:, 5] 
#             self.batch_ll_l2_length = self.batch_ll_l2_length.unsqueeze(1)
#             self.batch_ll_l2_length = self.batch_ll_l2_length.repeat(1, self.num_frames) # [B, F]
#         elif length_type == 'frame':
#             batch_init_limb_lens = batch_limb_lens[:, ref_frame, :] # [B, 16]
#             self.batch_ra_l1_length = batch_init_limb_lens[:, 14]
#             self.batch_ra_l1_length = self.batch_ra_l1_length.unsqueeze(1)
#             self.batch_ra_l1_length = self.batch_ra_l1_length.repeat(1, self.num_frames) # [B, F]
#             self.batch_ra_l2_length = batch_init_limb_lens[:, 15]
#             self.batch_ra_l2_length = self.batch_ra_l2_length.unsqueeze(1)
#             self.batch_ra_l2_length = self.batch_ra_l2_length.repeat(1, self.num_frames) # [B, F]
#             self.batch_la_l1_length = batch_init_limb_lens[:, 11]
#             self.batch_la_l1_length = self.batch_la_l1_length.unsqueeze(1)
#             self.batch_la_l1_length = self.batch_la_l1_length.repeat(1, self.num_frames) # [B, F]
#             self.batch_la_l2_length = batch_init_limb_lens[:, 12]
#             self.batch_la_l2_length = self.batch_la_l2_length.unsqueeze(1)
#             self.batch_la_l2_length = self.batch_la_l2_length.repeat(1, self.num_frames) # [B, F]
#             self.batch_rl_l1_length = batch_init_limb_lens[:, 1] 
#             self.batch_rl_l1_length = self.batch_rl_l1_length.unsqueeze(1)
#             self.batch_rl_l1_length = self.batch_rl_l1_length.repeat(1, self.num_frames) # [B, F]
#             self.batch_rl_l2_length = batch_init_limb_lens[:, 2] 
#             self.batch_rl_l2_length = self.batch_rl_l2_length.unsqueeze(1)
#             self.batch_rl_l2_length = self.batch_rl_l2_length.repeat(1, self.num_frames) # [B, F]
#             self.batch_ll_l1_length = batch_init_limb_lens[:, 4] 
#             self.batch_ll_l1_length = self.batch_ll_l1_length.unsqueeze(1)
#             self.batch_ll_l1_length = self.batch_ll_l1_length.repeat(1, self.num_frames) # [B, F]
#             self.batch_ll_l2_length = batch_init_limb_lens[:, 5] 
#             self.batch_ll_l2_length = self.batch_ll_l2_length.unsqueeze(1)
#             self.batch_ll_l2_length = self.batch_ll_l2_length.repeat(1, self.num_frames) # [B, F]
    
#     def update_batch_keypoints(self, torso_output):
#         '''
#             Input: (N, T, 9, 3)
#         '''
#         self.batch_pelvis     = torso_output[:, :, 0] # [B, F, 3]
#         self.batch_r_hip      = torso_output[:, :, 1] # [B, F, 3]
#         self.batch_l_hip      = torso_output[:, :, 2] # [B, F, 3]
#         self.batch_torso      = torso_output[:, :, 3] # [B, F, 3]
#         self.batch_neck       = torso_output[:, :, 4] # [B, F, 3]
#         self.batch_nose       = torso_output[:, :, 5] # [B, F, 3]
#         self.batch_head       = torso_output[:, :, 6] # [B, F, 3]
#         self.batch_l_shoulder = torso_output[:, :, 7] # [B, F, 3]
#         self.batch_r_shoulder = torso_output[:, :, 8] # [B, F, 3]
        
#     def update_body_reference_frame(self):
#         ## get body reference frame
#         # z axis
#         self.batch_z_axis = torch.tensor([0, 0, 1], dtype=torch.float32)
#         self.batch_z_axis = self.batch_z_axis.unsqueeze(0)
#         self.batch_z_axis = self.batch_z_axis.unsqueeze(0)
#         self.batch_z_axis = self.batch_z_axis.repeat(self.batch_size, self.num_frames, 1).cuda() # [B, F, 3]
#         # y axis
#         self.batch_y_axis = self.batch_l_hip - self.batch_pelvis # [B, F, 3]
#         self.batch_y_axis[:, :, 2] = 0
#         batch_y_axis_mag = torch.norm(self.batch_y_axis, dim=2)
#         batch_y_axis_mag = batch_y_axis_mag.unsqueeze(-1)
#         self.batch_y_axis /= batch_y_axis_mag
#         # x axis
#         self.batch_x_axis = torch.cross(self.batch_y_axis, self.batch_z_axis, dim=2) # [B, F, 3]
#         # body_R
#         A = self.batch_x_axis.unsqueeze(-1)
#         B = self.batch_y_axis.unsqueeze(-1)
#         C = self.batch_z_axis.unsqueeze(-1)
#         self.batch_body_R = torch.cat([A, B, C], dim=-1).transpose(2, 3) # [B, F, 3, 3]
        
#     def update_root_tf(self):
#         self.root_tf = torch.eye(4, dtype=torch.float32)
#         self.root_tf = self.root_tf.unsqueeze(0)
#         self.root_tf = self.root_tf.unsqueeze(0)
#         self.root_tf = self.root_tf.unsqueeze(0)
#         self.root_tf = self.root_tf.repeat(self.batch_size, self.num_frames, 4, 1, 1).cuda()
#         temp = torch.matmul(self.rot_y_180, self.batch_body_R.to(self.rot_y_180.device))
#         temp = temp.unsqueeze(2)
#         temp = temp.repeat(1, 1, 4, 1, 1) # [B, F, 4, 3, 3]
#         self.root_tf[:, :, :, :3, :3] = temp
#         self.root_tf[:, :, self.right_arm_id, :3, 3] = self.batch_r_shoulder # right arm
#         self.root_tf[:, :, self.left_arm_id , :3, 3] = self.batch_l_shoulder # left arm
#         self.root_tf[:, :, self.right_leg_id, :3, 3] = self.batch_r_hip # right leg
#         self.root_tf[:, :, self.left_leg_id , :3, 3] = self.batch_l_hip # left leg
        
#     def forward_appendage(self, appendage_id, batch_link1_azim, batch_link1_elev, batch_link2_azim, batch_link2_elev, batch_l1_length, batch_l2_length):
#         batch_link1_mat = self.batch_DH_matrix(batch_link1_azim, batch_link1_elev, self.batch_zero, self.batch_zero) # [B, F, 4, 4]
#         batch_link2_mat = self.batch_DH_matrix(batch_link2_azim, batch_link2_elev, self.batch_zero, batch_l1_length) # [B, F, 4, 4]
#         batch_terminal_mat = self.batch_DH_matrix(self.batch_zero, self.batch_zero, self.batch_zero, batch_l2_length) # [B, F, 4, 4]
        
#         batch_link1_tf = torch.matmul(self.root_tf[:, :, appendage_id, :, :], batch_link1_mat) # [B, F, 4, 4]
#         batch_link2_tf = torch.matmul(batch_link1_tf, batch_link2_mat) # [B, F, 4, 4]
#         batch_terminal_tf = torch.matmul(batch_link2_tf, batch_terminal_mat) # [B, F, 4, 4]
#         batch_link2_origin = batch_link2_tf[:, :, :3, 3] # [B, F, 3]
#         batch_terminal_origin = batch_terminal_tf[:, :, :3, 3] # [B, F, 3]
        
#         return batch_link2_origin, batch_terminal_origin
        
#     def get_pose_output(self):
#         pose_3d = torch.zeros(self.batch_size, self.num_frames, 17, 3, dtype=torch.float32).cuda()
#         pose_3d[:, :, 0]  = self.batch_pelvis 
#         pose_3d[:, :, 1]  = self.batch_r_hip
#         pose_3d[:, :, 2]  = self.batch_r_knee 
#         pose_3d[:, :, 3]  = self.batch_r_ankle
#         pose_3d[:, :, 4]  = self.batch_l_hip
#         pose_3d[:, :, 5]  = self.batch_l_knee 
#         pose_3d[:, :, 6]  = self.batch_l_ankle
#         pose_3d[:, :, 7]  = self.batch_torso
#         pose_3d[:, :, 8]  = self.batch_neck
#         pose_3d[:, :, 9]  = self.batch_nose
#         pose_3d[:, :, 10] = self.batch_head
#         pose_3d[:, :, 11] = self.batch_l_shoulder 
#         pose_3d[:, :, 12] = self.batch_l_elbow
#         pose_3d[:, :, 13] = self.batch_l_wrist
#         pose_3d[:, :, 14] = self.batch_r_shoulder
#         pose_3d[:, :, 15] = self.batch_r_elbow 
#         pose_3d[:, :, 16] = self.batch_r_wrist 
#         return pose_3d