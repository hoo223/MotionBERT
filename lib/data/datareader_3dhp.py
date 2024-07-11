# Adapted from Optimizing Network Structure for 3D Human Pose Estimation (ICCV 2019) (https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/data.py)

import numpy as np
import os, sys
import random
import copy
from lib.utils.tools import read_pkl
from lib.utils.utils_data import split_clips
random.seed(0)
    
class DataReader3DHP(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, dt_root = 'data/motion3d', dt_file = '3dhp_gt_testset.pkl', input_mode='joint_2d', gt_mode='joint3d_image'):
        self.gt_trainset = None
        self.gt_testset = None
        self.split_id_train = None
        self.split_id_test = None
        self.test_hw = None
        self.dt_dataset = read_pkl('%s/%s' % (dt_root, dt_file))
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence
        self.input_mode = input_mode
        self.gt_mode = gt_mode
        
    def read_2d(self):
        #print(self.dt_dataset['train']['joint_2d'].shape, self.dt_dataset['test']['joint_2d'].shape)
        trainset = self.dt_dataset['train'][self.input_mode][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        testset = self.dt_dataset['test'][self.input_mode][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        # map to [-1, 1]
        res_w, res_h = 2048, 2048
        trainset = trainset / res_w * 2 - [1, res_h / res_w]
        testset = (testset * 2 - [res_w, res_h]) / res_w
        if self.read_confidence:
            if 'confidence' in self.dt_dataset['train'].keys():
                train_confidence = self.dt_dataset['train']['confidence'][::self.sample_stride].astype(np.float32)  
                test_confidence = self.dt_dataset['test']['confidence'][::self.sample_stride].astype(np.float32)  
                if len(train_confidence.shape)==2: # (1559752, 17)
                    train_confidence = train_confidence[:,:,None]
                    test_confidence = test_confidence[:,:,None]
            else:
                # No conf provided, fill with 1.
                train_confidence = np.ones(trainset.shape)[:,:,0:1]
                test_confidence = np.ones(testset.shape)[:,:,0:1]
            print(trainset.shape, train_confidence.shape)
            trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]
        return trainset, testset

    def read_3d(self):
        train_labels = self.dt_dataset['train'][self.gt_mode][::self.sample_stride, :, :3].astype(np.float32)  # [N, 17, 3]
        test_labels = self.dt_dataset['test'][self.gt_mode][::self.sample_stride, :, :3].astype(np.float32)    # [N, 17, 3]
        if self.gt_mode == 'joint3d_image': # normalize to [-1, 1]
            # map to [-1, 1]
            res_w, res_h = 2048, 2048
            train_labels[:, :, :2] = train_labels[:, :, :2] / res_w * 2 - [1, res_h / res_w]
            train_labels[:, :, 2:] = train_labels[:, :, 2:] / res_w * 2
            test_labels[:, :, :2] = test_labels[:, :, :2] / res_w * 2 - [1, res_h / res_w]
            test_labels[:, :, 2:] = test_labels[:, :, 2:] / res_w * 2
        elif self.gt_mode == 'world_3d' or self.gt_mode == 'cam_3d' or self.gt_mode == 'cam_3d_from_canonical_3d':
            pass
        else:
            raise ValueError("Invalid mode for read_3d: {}".format(self.gt_mode))
            
        return train_labels, test_labels
    
    def read_hw(self):
        if self.test_hw is not None:
            return self.test_hw
        test_hw = np.ones((len(self.dt_dataset['test']['camera_name']), 2)) * 2048
        self.test_hw = test_hw
        return test_hw
    
    def get_split_id(self):
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]                          # (1559752,)
        vid_list_test = self.dt_dataset['test']['source'][::self.sample_stride]                           # (566920,)
        self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train) 
        self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)
        return self.split_id_train, self.split_id_test
    
    def get_hw(self):
#       Only Testset HW is needed for denormalization
        test_hw = self.read_hw()                                     # train_data (1559752, 2) test_data (566920, 2)
        split_id_train, split_id_test = self.get_split_id()
        test_hw = test_hw[split_id_test][:,0,:]                      # (N, 2)
        return test_hw
    
    def get_sliced_data(self):
        train_data, test_data = self.read_2d()     # train_data (1559752, 17, 3) test_data (566920, 17, 3)
        train_labels, test_labels = self.read_3d() # train_labels (1559752, 17, 3) test_labels (566920, 17, 3)
        split_id_train, split_id_test = self.get_split_id()
        train_data, test_data = train_data[split_id_train], test_data[split_id_test]                # (N, 27, 17, 3)
        train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]        # (N, 27, 17, 3)
        # ipdb.set_trace()
        return train_data, test_data, train_labels, test_labels
    
    def denormalize(self, test_data):
        # test_data: (N, n_frames, 51) or test_data: (N, n_frames, 17, 3)        
        if self.gt_mode == 'joint3d_image':
            n_clips = test_data.shape[0]
            test_hw = self.get_hw()
            num_keypoints = test_data.shape[2]
            data = test_data.reshape([n_clips, -1, num_keypoints, 3])
            #assert len(data) == len(test_hw), "len(data): {}, len(test_hw): {}".format(len(data), len(test_hw))
            # denormalize (x,y,z) coordiantes for results
            for idx, item in enumerate(data):
                res_w, res_h = test_hw[idx]
                data[idx, :, :, :2] = (data[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
                data[idx, :, :, 2:] = data[idx, :, :, 2:] * res_w / 2
            return data # [n_clips, -1, 17, 3]
        elif self.gt_mode == 'world_3d' or self.gt_mode == 'cam_3d' or self.gt_mode == 'cam_3d_from_canonical_3d':
            return test_data
        else:
            raise ValueError("Invalid mode: {}".format(self.gt_mode))
