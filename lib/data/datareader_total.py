# Adapted from Optimizing Network Structure for 3D Human Pose Estimation (ICCV 2019) (https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/data.py)
import os, sys, getpass
user = getpass.getuser()
sys.path.append(f'/home/{user}/codes/hpe_library')
from hpe_library.lib_import import *
from hpe_library. my_utils import *

from lib.utils.tools import read_pkl
from lib.utils.utils_data import split_clips
random.seed(0)

class DataReaderTotal(object):
    def __init__(self, n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243, read_confidence=True, normalize_2d=True,
                 yaml_root='data/motion3d/yaml_files',
                 subset='',
                 step_rot=0,
                 overwrite_list = [],
                 default_data_type_list = ['source', 'cam_param', 'camera_name', 'action', 'confidence'],
                 source_tag='',
                 verbose=True):
        # load yaml file
        assert subset != '', 'subset should be provided'
        yaml_path = os.path.join(yaml_root, subset+'.yaml')
        if not os.path.exists(yaml_path):
            print(f'{yaml_path} does not exist... generating yaml file')
            gernerate_dataset_yaml(subset)
        with open(os.path.join(yaml_path), 'r') as file:
            self.yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        self.dataset_name   = self.yaml_data['dataset_name']
        self.default_data_type_list = default_data_type_list
        self.data_type_list = self.yaml_data['data_type_list'] + default_data_type_list
        self.canonical_type = self.yaml_data['canonical_type']
        self.input_source   = self.yaml_data['input_source']
        self.input_mode     = self.yaml_data['input_mode']
        self.gt_mode        = self.yaml_data['gt_mode']
        self.mpjpe_mode     = self.yaml_data['mpjpe_mode']
        self.univ           = self.yaml_data['univ']
        try:
            self.train_subject  = self.yaml_data['train_subject']
        except:
            self.train_subject  = []
        self.test_subject   = self.yaml_data['test_subject']
        self.train_cam      = self.yaml_data['train_cam']
        self.test_cam       = self.yaml_data['test_cam']
        self.cam_list       = self.yaml_data['cam_list']
        self.adaptive_focal = self.yaml_data['adaptive_focal']
        self.data_aug = {'step_rot': self.yaml_data['step_rot'],
                         'sinu_yaw_mag': self.yaml_data['sinu_yaw_mag'],
                         'sinu_yaw_period': self.yaml_data['sinu_yaw_period'],
                         'sinu_pitch_mag': self.yaml_data['sinu_pitch_mag'],
                         'sinu_pitch_period': self.yaml_data['sinu_pitch_period'],
                         'sinu_roll_mag': self.yaml_data['sinu_roll_mag'],
                         'sinu_roll_period': self.yaml_data['sinu_roll_period'],
                         'rand_yaw_mag': self.yaml_data['rand_yaw_mag'],
                         'rand_yaw_period': self.yaml_data['rand_yaw_period'],
                         'rand_pitch_mag': self.yaml_data['rand_pitch_mag'],
                         'rand_pitch_period': self.yaml_data['rand_pitch_period'],
                         'rand_roll_mag': self.yaml_data['rand_roll_mag'],
                         'rand_roll_period': self.yaml_data['rand_roll_period'],
                        }

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
        self.source_tag = source_tag
        self.normalize_2d = normalize_2d

        print(f"Loading data type: {self.data_type_list}")
        assert self.input_mode in self.data_type_list, f'{self.input_mode} should be in data_type_list {self.data_type_list}'
        assert self.gt_mode in self.data_type_list, f'{self.gt_mode} should be in data_type_list {self.data_type_list}'
        assert self.mpjpe_mode in self.data_type_list, f'{self.mpjpe_mode} should be in data_type_list {self.data_type_list}'

        self.dt_dataset = self.generate_total_dataset(verbose)

    def generate_total_dataset(self, verbose):
        dt_dataset = {'train': {}, 'test': {}}
        source_list = load_data(dataset_name=self.dataset_name, data_type='source_list', overwrite_list=self.overwrite_list, verbose=verbose)
        cam_params = load_data(dataset_name=self.dataset_name, data_type='cam_param', adaptive_focal=self.adaptive_focal, overwrite_list=self.overwrite_list, verbose=verbose)

        for data_type in self.data_type_list:
            # type mapping
            if data_type   == 'cam_3d_from_canonical_3d':        load_type = 'cam_3d_canonical'
            elif data_type == 'joint_2d':                        load_type = 'img_2d'
            elif data_type == 'joint_2d_from_canonical_3d':      load_type = 'img_2d_canonical'
            elif data_type == 'joint3d_image':                   load_type = 'img_3d'
            elif data_type == 'joint3d_image_from_canonical_3d': load_type = 'img_3d_canonical'
            elif data_type == '2.5d_factor':                     load_type = 'scale_factor'
            elif data_type == '2.5d_factor_from_canonical_3d':   load_type = 'scale_factor_canonical'
            elif data_type == 'joints_2.5d_image':               load_type = 'img_25d'
            else:                                                load_type = data_type
            # load data
            if data_type not in ['source', 'cam_param', 'camera_name', 'action', 'confidence']:
                data = load_data(dataset_name=self.dataset_name, data_type=load_type, canonical_type=self.canonical_type, univ=self.univ, adaptive_focal=self.adaptive_focal, data_aug=self.data_aug, overwrite_list=self.overwrite_list, verbose=verbose)
            # initialize dt_dataset
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
                if data_type == 'source':
                    if self.source_tag == '': dt_dataset[train_type][data_type] += [source]*num_frames
                    else: dt_dataset[train_type][data_type] += ['_'.join([source, self.source_tag])]*num_frames
                elif data_type == 'cam_param': dt_dataset[train_type][data_type] += [cam_params[subject][action][cam_id]]*num_frames
                elif data_type == 'camera_name': dt_dataset[train_type][data_type] += [cam_id]*num_frames
                elif data_type == 'action': dt_dataset[train_type][data_type] += [action]*num_frames
                elif data_type == 'confidence': dt_dataset[train_type][data_type] += [np.ones(17)]*num_frames
                elif data_type == 'world_3d': dt_dataset[train_type][data_type] += list(data[subject][action])
                else: dt_dataset[train_type][data_type] += list(data[subject][action][cam_id])
            for train_type in ['train', 'test']:
                dt_dataset[train_type][data_type] = np.array(dt_dataset[train_type][data_type])
            if len(self.train_subject) == 0:
                dt_dataset['train'][data_type] = dt_dataset['test'][data_type][:self.data_stride_test].copy()
            assert len(dt_dataset['train'][data_type]) > 0, f'{data_type} should not be empty (length: {len(dt_dataset["train"][data_type])})'
            assert len(dt_dataset['test'][data_type]) > 0, f'{data_type} should not be empty (length: {len(dt_dataset["test"][data_type])})'

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
        if self.normalize_2d:
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
        # normalize to [-1, 1]
        train_cam_param = self.dt_dataset['train']['cam_param'][::self.sample_stride]
        test_cam_param = self.dt_dataset['test']['cam_param'][::self.sample_stride]
        train_W, train_H = np.array([cam_param['W'] for cam_param in train_cam_param]), np.array([cam_param['H'] for cam_param in train_cam_param]) # (N,), (N,)
        test_W, test_H = np.array([cam_param['W'] for cam_param in test_cam_param]), np.array([cam_param['H'] for cam_param in test_cam_param]) # (N,), (N,)
        if self.gt_mode == 'joint3d_image': # normalize to [-1, 1]
            train_labels = self.normalize(train_labels, train_W, train_H, mode='3d')
            test_labels = self.normalize(test_labels, test_W, test_H, mode='3d')
        elif self.gt_mode in ['world_3d', 'cam_3d', 'cam_3d_from_canonical_3d', 'img_3d_norm', 'img_3d_norm_canonical']:
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
        test_hw = np.zeros((len(self.dt_dataset['test']['camera_name']), 2)) # (total_frame_num, 2)
        for idx, cam_param in enumerate(self.dt_dataset['test']['cam_param']):
            res_w, res_h = cam_param['W'], cam_param['H']
            test_hw[idx] = res_w, res_h
        self.test_hw = test_hw
        return test_hw

    def read_cam_param(self):
        train_cam_param = self.dt_dataset['train']['cam_param'][::self.sample_stride]
        test_cam_param = self.dt_dataset['test']['cam_param'][::self.sample_stride]
        return train_cam_param, test_cam_param

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

    def get_sliced_cam_param(self):
        train_cam_param, test_cam_param = self.read_cam_param()
        split_id_train, split_id_test = self.get_split_id()
        train_cam_param, test_cam_param = train_cam_param[split_id_train], test_cam_param[split_id_test]
        return train_cam_param, test_cam_param

    def get_sliced_data(self, with_cam_param=False):
        train_data, test_data = self.read_2d()     # train_data (1559752, 17, 3) test_data (566920, 17, 3)
        train_labels, test_labels = self.read_3d() # train_labels (1559752, 17, 3) test_labels (566920, 17, 3)
        print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
        split_id_train, split_id_test = self.get_split_id()
        train_data, test_data = train_data[split_id_train], test_data[split_id_test]                # (N, 27, 17, 3)
        train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]        # (N, 27, 17, 3)
        # if with_cam_param:
        #     train_cam_param, test_cam_param = self.read_cam_param()
        #     train_cam_param, test_cam_param = train_cam_param[split_id_train], test_cam_param[split_id_test]
        #     return train_data, test_data, train_labels, test_labels, train_cam_param, test_cam_param
        return train_data, test_data, train_labels, test_labels

    def denormalize(self, test_data):
        # test_data: (N, F, 51) or test_data: (N, F, 17, 3)
        if self.gt_mode == 'joint3d_image':
            n_clips = test_data.shape[0] # N
            test_hw = self.get_hw()
            num_keypoints = test_data.shape[2]
            data = test_data.reshape([n_clips, -1, num_keypoints, 3])
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

    def get_clip_info(self, args, num_results_all):
        _, split_id_test = self.get_split_id() # [range(0, 243) ... range(102759, 103002)]
        actions = np.array(self.dt_dataset['test']['action']) # 103130 ['squat' ...  'kneeup']
        if args.scale_consistency:
            if 'canonical' in self.gt_mode: factors = np.array(self.dt_dataset['test']['scale_factor_norm_canonical'])
            else:                           factors = np.array(self.dt_dataset['test']['scale_factor_norm'])
        else:
            try:    factors = np.array(self.dt_dataset['test']['2.5d_factor']) # 103130 [3.49990559 ... 2.09230852]
            except: factors = np.ones_like(actions) # if no factor
        gts = np.array(self.dt_dataset['test'][args.mpjpe_mode])
        sources = np.array(self.dt_dataset['test']['source']) # 103130 ['S02_6_squat_001' ... 'S08_4_kneeup_001']

        num_test_frames = len(actions)
        frames = np.array(range(num_test_frames))
        action_clips = np.array([actions[split_id_test[i]] for i in range(len(split_id_test))]) # actions[split_id_test]
        factor_clips = np.array([factors[split_id_test[i]] for i in range(len(split_id_test))]) # factors[split_id_test]
        source_clips = np.array([sources[split_id_test[i]] for i in range(len(split_id_test))]) # sources[split_id_test]
        frame_clips  = np.array([frames[split_id_test[i]] for i in range(len(split_id_test))]) # frames[split_id_test]
        gt_clips     = np.array([gts[split_id_test[i]] for i in range(len(split_id_test))]) # gts[split_id_test]

        action_clips = action_clips[:num_results_all]
        factor_clips = factor_clips[:num_results_all]
        source_clips = source_clips[:num_results_all]
        frame_clips = frame_clips[:num_results_all]
        gt_clips = gt_clips[:num_results_all]
        assert num_results_all==len(action_clips), f'The number of results and action_clips are different: {num_results_all} vs {len(action_clips)}'

        return num_test_frames, action_clips, factor_clips, source_clips, frame_clips, gt_clips, actions

    def get_action_list(self):
        return sorted(set(self.dt_dataset['test']['action']))

class DataReaderTotalGroup(object):
    def __init__(self, n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243, read_confidence=True, normalize_2d=True,
                 yaml_root='data/motion3d/yaml_files',
                 subset_list=[],
                 overwrite_list = [],
                 default_data_type_list = ['source', 'cam_param', 'camera_name', 'action', 'confidence'],
                 verbose=True):
        self.channel_pose_2d = 3 if read_confidence else 2
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence
        self.normalize_2d = normalize_2d
        self.split_id_train = None
        self.split_id_test  = None
        self.test_hw        = None
        self.datareader = {}
        assert len(subset_list) > 0, 'subset_list should be provided'
        for i, subset in enumerate(subset_list):
            print(subset)
            self.datareader[subset] = copy.deepcopy(DataReaderTotal(n_frames=n_frames, sample_stride=sample_stride, data_stride_train=data_stride_train, data_stride_test=data_stride_test, read_confidence=read_confidence, normalize_2d=normalize_2d,
                                                      yaml_root=yaml_root, subset=subset, overwrite_list=overwrite_list, default_data_type_list=default_data_type_list, verbose=verbose))

    def normalize(self, data, W_array, H_array, mode):
        raise NotImplementedError('normalize method should be implemented')

    def read_2d(self):
        total_train_data, total_test_data = np.empty([0, 17, self.channel_pose_2d]), np.empty([0, 17, self.channel_pose_2d])
        num_train_data, num_test_data = 0, 0
        for subset in self.datareader.keys():
            train_data, test_data = self.datareader[subset].read_2d()
            #print('2d data shape', train_data.shape, test_data.shape)
            num_train_data += train_data.shape[0]
            num_test_data += test_data.shape[0]
            total_train_data = np.concatenate([total_train_data, train_data], axis=0)
            total_test_data = np.concatenate([total_test_data, test_data], axis=0)
        # check number of data
        assert total_train_data.shape[0] == num_train_data, f'{total_train_data.shape[0]} {num_train_data}'
        assert total_test_data.shape[0] == num_test_data, f'{total_test_data.shape[0]} {num_test_data}'
        return total_train_data, total_test_data

    def read_3d(self):
        total_train_labels, total_test_labels = np.empty([0, 17, 3]), np.empty([0, 17, 3])
        num_train_labels, num_test_labels = 0, 0
        for subset in self.datareader.keys():
            train_labels, test_labels = self.datareader[subset].read_3d()
            num_train_labels += train_labels.shape[0]
            num_test_labels += test_labels.shape[0]
            total_train_labels = np.concatenate([total_train_labels, train_labels], axis=0)
            total_test_labels = np.concatenate([total_test_labels, test_labels], axis=0)
        # check number of data
        assert total_train_labels.shape[0] == num_train_labels, f'{total_train_labels.shape[0]} {num_train_labels}'
        assert total_test_labels.shape[0] == num_test_labels, f'{total_test_labels.shape[0]} {num_test_labels}'
        return total_train_labels, total_test_labels

    def read_cam_param(self):
        total_train_cam_param, total_test_cam_param = np.empty([0]), np.empty([0])
        num_train_cam_param, num_test_cam_param = 0, 0
        for subset in self.datareader.keys():
            train_cam_param, test_cam_param = self.datareader[subset].read_cam_param()
            num_train_cam_param += train_cam_param.shape[0]
            num_test_cam_param += test_cam_param.shape[0]
            total_train_cam_param = np.concatenate([total_train_cam_param, train_cam_param], axis=0)
            total_test_cam_param = np.concatenate([total_test_cam_param, test_cam_param], axis=0)
        # check number of data
        assert total_train_cam_param.shape[0] == num_train_cam_param, f'{total_train_cam_param.shape[0]} {num_train_cam_param}'
        assert total_test_cam_param.shape[0] == num_test_cam_param, f'{total_test_cam_param.shape[0]} {num_test_cam_param}'
        return total_train_cam_param, total_test_cam_param

    def read_hw(self):
        total_test_hw = np.empty([0, 2])
        num_test_hw = 0
        for subset in self.datareader.keys():
            test_hw = self.datareader[subset].read_hw()
            num_test_hw += test_hw.shape[0]
            total_test_hw = np.concatenate([total_test_hw, test_hw], axis=0)
        # check number of data
        assert total_test_hw.shape[0] == num_test_hw, f'{total_test_hw.shape[0]} {num_test_hw}'
        return total_test_hw

    def get_split_id(self):
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train, vid_list_test = np.empty([0]), np.empty([0])
        for subset in self.datareader.keys():
            vid_list_train = np.concatenate([vid_list_train, self.datareader[subset].dt_dataset['train']['source'][::self.datareader[subset].sample_stride]], axis=0) # (1559752,)
            vid_list_test = np.concatenate([vid_list_test, self.datareader[subset].dt_dataset['test']['source'][::self.datareader[subset].sample_stride]], axis=0)  # (566920,)
        self.vid_list_train = vid_list_train
        self.vid_list_test = vid_list_test
        #print('vid list shape', vid_list_train.shape, vid_list_test.shape)
        self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train)
        self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)
        #print('spilt id length', len(self.split_id_train), len(self.split_id_test))

        return self.split_id_train, self.split_id_test

    def get_hw(self):
        # Only Testset HW is needed for denormalization
        test_hw = self.read_hw()                                     # train_data (1559752, 2) test_data (566920, 2)
        split_id_train, split_id_test = self.get_split_id()
        test_hw = test_hw[split_id_test][:,0,:]                      # (N, 2)
        return test_hw

    def get_sliced_data(self, with_cam_param=False):
        train_data, test_data = self.read_2d()     # train_data (1559752, 17, 3) test_data (566920, 17, 3)
        train_labels, test_labels = self.read_3d() # train_labels (1559752, 17, 3) test_labels (566920, 17, 3)
        #print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
        split_id_train, split_id_test = self.get_split_id()
        train_data, test_data = train_data[split_id_train], test_data[split_id_test]                # (N, 27, 17, 3)
        train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]        # (N, 27, 17, 3)
        # if with_cam_param:
        #     train_cam_param, test_cam_param = self.read_cam_param()
        #     train_cam_param, test_cam_param = train_cam_param[split_id_train], test_cam_param[split_id_test]
        #     return train_data, test_data, train_labels, test_labels, train_cam_param, test_cam_param
        return train_data, test_data, train_labels, test_labels

    def get_sliced_cam_param(self):
        train_cam_param, test_cam_param = self.read_cam_param()
        split_id_train, split_id_test = self.get_split_id()
        train_cam_param, test_cam_param = train_cam_param[split_id_train], test_cam_param[split_id_test]
        return train_cam_param, test_cam_param

    def denormalize(self, test_data):
        pre_len = 0
        denormalized = np.empty([0, self.n_frames, 17, 3])
        for subset in self.datareader.keys():
            test_len = len(self.datareader[subset].dt_dataset['test'][self.datareader[subset].gt_mode][::self.datareader[subset].sample_stride])
            #print(pre_len, pre_len+test_len)
            #print(denormalized.shape, self.datareader[subset].denormalize(test_data[pre_len:pre_len+test_len]).shape)
            denormalized = np.concatenate([denormalized, self.datareader[subset].denormalize(test_data[pre_len:pre_len+test_len])], axis=0)
            pre_len += test_len
        return denormalized

    def get_clip_info(self, args, num_results_all):
        _, split_id_test = self.get_split_id() # [range(0, 243) ... range(102759, 103002)]
        total_actions = np.empty([0])
        total_factors = np.empty([0])
        total_sources = np.empty([0])
        total_gts = np.empty([0, 17, 3])
        for subset in self.datareader.keys():
            total_actions = np.concatenate([total_actions, self.datareader[subset].dt_dataset['test']['action']], axis=0)
            if args.scale_consistency:
                if 'canonical' in self.datareader[subset].gt_mode: total_factors = np.concatenate([total_factors, self.datareader[subset].dt_dataset['test']['scale_factor_norm_canonical']], axis=0)
                else:                                               total_factors = np.concatenate([total_factors, self.datareader[subset].dt_dataset['test']['scale_factor_norm']], axis=0)
            else:
                try: total_factors = np.concatenate([total_factors, self.datareader[subset].dt_dataset['test']['2.5d_factor']], axis=0)
                except: total_factors = np.ones_like(total_actions)
            total_sources = np.concatenate([total_sources, self.datareader[subset].dt_dataset['test']['source']], axis=0)
            total_gts = np.concatenate([total_gts, self.datareader[subset].dt_dataset['test'][args.mpjpe_mode]], axis=0)

        num_test_frames = len(total_actions)
        total_frames = np.array(range(num_test_frames))
        action_clips = np.array([total_actions[split_id_test[i]] for i in range(len(split_id_test))]) # actions[split_id_test]
        factor_clips = np.array([total_factors[split_id_test[i]] for i in range(len(split_id_test))]) # factors[split_id_test]
        source_clips = np.array([total_sources[split_id_test[i]] for i in range(len(split_id_test))]) # sources[split_id_test]
        frame_clips  = np.array([total_frames[split_id_test[i]] for i in range(len(split_id_test))]) # frames[split_id_test]
        gt_clips     = np.array([total_gts[split_id_test[i]] for i in range(len(split_id_test))]) # gts[split_id_test]

        action_clips = action_clips[:num_results_all]
        factor_clips = factor_clips[:num_results_all]
        source_clips = source_clips[:num_results_all]
        frame_clips = frame_clips[:num_results_all]
        gt_clips = gt_clips[:num_results_all]
        assert num_results_all==len(action_clips), f'The number of results and action_clips are different: {num_results_all} vs {len(action_clips)}'

        return num_test_frames, action_clips, factor_clips, source_clips, frame_clips, gt_clips, total_actions

    def get_action_list(self):
        action_list = []
        for subset in self.datareader.keys():
            action_list += self.datareader[subset].get_action_list()
        return sorted(set(action_list))