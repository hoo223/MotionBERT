# Adapted from Optimizing Network Structure for 3D Human Pose Estimation (ICCV 2019) (https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/data.py)
import os, sys, getpass
user = getpass.getuser()
sys.path.append(f'/home/{user}/codes/hpe_library')
from lib_import import *
from my_utils import *

from lib.utils.tools import read_pkl
from lib.utils.utils_data import split_clips
random.seed(0)
    
class DataReaderTotal(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, 
                 yaml_root='data/motion3d/MB3D_f243s81', 
                 subset='', 
                 overwrite_list = [],
                 default_data_type_lsit = ['source', 'cam_param', 'camera_name', 'action', 'confidence'],
                 verbose=True):
        yaml_path = os.path.join(yaml_root, subset+'.yaml')
        assert os.path.exists(yaml_path), f'{yaml_path} does not exist'
        with open(os.path.join(yaml_path), 'r') as file:
            self.yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        self.dataset_name   = self.yaml_data['dataset_name']
        self.data_type_list = self.yaml_data['data_type_list'] + default_data_type_lsit
        self.canonical_type = self.yaml_data['canonical_type']
        self.input_source   = self.yaml_data['input_source']
        self.input_mode     = self.yaml_data['input_mode']
        self.gt_mode        = self.yaml_data['gt_mode']
        self.train_subject  = self.yaml_data['train_subject']
        self.test_subject   = self.yaml_data['test_subject']
        self.train_cam      = self.yaml_data['train_cam']
        self.test_cam       = self.yaml_data['test_cam']
        self.cam_list       = self.yaml_data['cam_list']
        self.default_data_type_lsit = default_data_type_lsit
        self.overwrite_list = overwrite_list
        self.gt_trainset    = None
        self.gt_testset     = None
        self.split_id_train = None
        self.split_id_test  = None
        self.test_hw        = None
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence
        
        assert self.input_mode in self.data_type_list, f'{self.input_mode} should be in data_type_list {self.data_type_list}'
        assert self.gt_mode in self.data_type_list, f'{self.gt_mode} should be in data_type_list {self.data_type_list}'

        self.dt_dataset = self.generate_total_dataset(verbose)
        
    def generate_total_dataset(self, verbose):
        dt_dataset = {'train': {}, 'test': {}}
        source_list = load_data(dataset_name=self.dataset_name, data_type='source_list', overwrite_list=self.overwrite_list, verbose=verbose)
        cam_params = load_data(dataset_name=self.dataset_name, data_type='cam_param', overwrite_list=self.overwrite_list, verbose=verbose)
        
        for data_type in self.data_type_list:
            # type mapping
            if data_type == 'cam_3d_from_canonical_3d':          load_type = 'cam_3d_canonical'
            elif data_type == 'joint_2d':                        load_type = 'img_2d'
            elif data_type == 'joint_2d_from_canonical_3d':      load_type = 'img_2d_canonical'         
            elif data_type == 'joint3d_image_from_canonical_3d': load_type = 'img_3d_canonical'
            elif data_type == '2.5d_factor':                     load_type = 'scale_factor'
            elif data_type == '2.5d_factor_from_canonical_3d':   load_type = 'scale_factor_canonical'
            else:                                                load_type = data_type
            # load data
            if data_type not in ['source', 'cam_param', 'camera_name', 'action', 'confidence']:
                data = load_data(dataset_name=self.dataset_name, data_type=load_type, canonical_type=self.canonical_type, overwrite_list=self.overwrite_list, verbose=verbose)
            # initialize
            for train_type in ['train', 'test']:
                dt_dataset[train_type][data_type] = []
            # assign data
            for source in source_list:
                subject, cam_id, action = split_source_name(source, self.dataset_name)
                if len(self.cam_list) > 0 and cam_id not in self.cam_list: continue
                
                if self.train_subject != self.test_subject:
                    if subject in self.test_subject: train_type = 'test'
                    elif subject in self.train_subject: train_type = 'train'
                    else: continue
                else: 
                    if cam_id in self.train_cam: train_type = 'train'
                    elif cam_id in self.test_cam: train_type = 'test'
                    else: continue
                
                num_frames = cam_params[subject][action][cam_id]['num_frames']
                if data_type == 'source': dt_dataset[train_type][data_type] += [source]*num_frames
                elif data_type == 'cam_param': dt_dataset[train_type][data_type] += [cam_params[subject][action][cam_id]]*num_frames
                elif data_type == 'camera_name': dt_dataset[train_type][data_type] += [cam_id]*num_frames
                elif data_type == 'action': dt_dataset[train_type][data_type] += [action]*num_frames
                elif data_type == 'confidence': dt_dataset[train_type][data_type] += [np.ones(17)]*num_frames
                elif data_type == 'world_3d': dt_dataset[train_type][data_type] += list(data[subject][action])
                else: 
                    dt_dataset[train_type][data_type] += list(data[subject][action][cam_id])
            for train_type in ['train', 'test']:
                dt_dataset[train_type][data_type] = np.array(dt_dataset[train_type][data_type])
            if len(self.train_subject) == 0:
                dt_dataset['train'][data_type] = dt_dataset['test'][data_type][:self.data_stride_test].copy()
            assert len(dt_dataset['train'][data_type]) > 0, f'{data_type} should not be empty'
            assert len(dt_dataset['test'][data_type]) > 0, f'{data_type} should not be empty'
            
        return dt_dataset 
        
    def normalize(self, data, W_array, H_array, mode):
        assert len(data) == len(W_array), f'data, W_array should have the same length {len(data)} {len(W_array)}'
        assert len(data) == len(H_array), f'data, H_array should have the same length {len(data)} {len(H_array)}'
        if mode == '2d':
            data *= 2
            data -= np.concatenate([W_array[:, None, None], H_array[:, None, None]], axis=2)
            data /= W_array[:, None, None]
        elif mode == '3d':
            data *= 2
            data[...,:2] -= np.concatenate([W_array[:, None, None], H_array[:, None, None]], axis=2)
            data /= W_array[:, None, None]
        else:
            raise ValueError('Invalid mode')
        return data
        
    def read_2d(self):
        trainset = self.dt_dataset['train'][self.input_mode][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        testset = self.dt_dataset['test'][self.input_mode][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        train_cam_param = self.dt_dataset['train']['cam_param'][::self.sample_stride]
        test_cam_param = self.dt_dataset['test']['cam_param'][::self.sample_stride]
        train_W, train_H = np.array([cam_param['W'] for cam_param in train_cam_param]), np.array([cam_param['H'] for cam_param in train_cam_param])
        test_W, test_H = np.array([cam_param['W'] for cam_param in test_cam_param]), np.array([cam_param['H'] for cam_param in test_cam_param])
        # normalize to [-1, 1]
        trainset = self.normalize(trainset, train_W, train_H, mode='2d')
        testset = self.normalize(testset, test_W, test_H, mode='2d')
        # add confidence
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
            trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]
        return trainset, testset

    def read_3d(self):
        train_labels = self.dt_dataset['train'][self.gt_mode][::self.sample_stride, :, :3].astype(np.float32)  # [N, 17, 3]
        test_labels = self.dt_dataset['test'][self.gt_mode][::self.sample_stride, :, :3].astype(np.float32)    # [N, 17, 3]
        train_cam_param = self.dt_dataset['train']['cam_param'][::self.sample_stride]
        test_cam_param = self.dt_dataset['test']['cam_param'][::self.sample_stride]
        train_W, train_H = np.array([cam_param['W'] for cam_param in train_cam_param]), np.array([cam_param['H'] for cam_param in train_cam_param]) # (N,), (N,)
        test_W, test_H = np.array([cam_param['W'] for cam_param in test_cam_param]), np.array([cam_param['H'] for cam_param in test_cam_param]) # (N,), (N,)
        if self.gt_mode == 'joint3d_image': # normalize to [-1, 1]
            # map to [-1, 1]
            train_labels = self.normalize(train_labels, train_W, train_H, mode='3d')
            test_labels = self.normalize(test_labels, test_W, test_H, mode='3d')
        elif self.gt_mode == 'world_3d' or self.gt_mode == 'cam_3d' or self.gt_mode == 'cam_3d_from_canonical_3d':
            pass
        elif self.gt_mode == 'joint_2d_from_canonical_3d':
            temp_input_mode = self.input_mode
            self.input_mode = 'joint_2d_from_canonical_3d'
            self.input_mode = temp_input_mode
        else:
            raise ValueError("Invalid mode for read_3d: {}".format(self.gt_mode))
            
        return train_labels, test_labels
    
    def read_hw(self):
        if self.test_hw is not None:
            return self.test_hw
        test_hw = np.zeros((len(self.dt_dataset['test']['camera_name']), 2))
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            test_hw[idx] = res_w, res_h
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
        # Only Testset HW is needed for denormalization
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
            raise ValueError("Invalid mode for denormalize: {}".format(self.gt_mode))