from torch.utils.data import DataLoader

from lib.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D
from lib.data.dataset_motion_3d import MotionDataset3D, MotionDataset3DTotal
from lib.data.augmentation import Augmenter2D
from lib.data.datareader_h36m import DataReaderH36M
from lib.data.datareader_aihub import DataReaderAIHUB
from lib.data.datareader_fit3d import DataReaderFIT3D
from lib.data.datareader_kookmin import DataReaderKOOKMIN
from lib.data.datareader_3dhp import DataReader3DHP
from lib.data.datareader_poseaug_3dhp import DataReaderPOSEAUG3DHP
from lib.data.datareader_total import DataReaderTotal, DataReaderTotalGroup

def load_dataset(args, use_new_datareader=False):
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

    if args.train_2d:
        posetrack = PoseTrackDataset2D()
        posetrack_loader_2d = DataLoader(posetrack, **trainloader_params)
        instav = InstaVDataset2D()
        instav_loader_2d = DataLoader(instav, **trainloader_params)
    else:
        posetrack_loader_2d = None
        instav_loader_2d = None

    if use_new_datareader:
        if len(args.subset_list) >= 1:
            datareader = DataReaderTotalGroup(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, subset_list=args.subset_list)
        # elif len(args.subset_list) == 1:
        #     subset = args.subset_list[0]
        #     datareader = DataReaderTotal(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, subset=subset, step_rot=args.step_rot)
        else:
            raise ValueError('Subset list is empty')
        train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()
        train_dataset = MotionDataset3DTotal(args, train_data.copy(), train_labels.copy(), 'train')
        test_dataset = MotionDataset3DTotal(args, test_data.copy(), test_labels.copy(), 'test')
    else:
        for subset in args.subset_list:
            print(subset)
            if 'H36M' in subset:
                datareader = DataReaderH36M(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = 'data/motion3d', dt_file=args.dt_file, input_mode=args.input_mode, gt_mode=args.gt_mode)
            elif 'AIHUB'   in subset: datareader = DataReaderAIHUB(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = 'data/motion3d', dt_file=args.dt_file)
            elif 'FIT3D'   in subset: datareader = DataReaderFIT3D(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = 'data/motion3d', dt_file=args.dt_file, input_mode=args.input_mode, gt_mode=args.gt_mode)
            elif 'KOOKMIN' in subset: datareader = DataReaderKOOKMIN(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = 'data/motion3d', dt_file=args.dt_file)
            elif '3DHP' in subset:
                if 'POSEAUG' in subset: datareader = DataReaderPOSEAUG3DHP(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = 'data/motion3d', dt_file=args.dt_file, mode=args.gt_mode)
                else:                   datareader = DataReader3DHP(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = 'data/motion3d', dt_file=args.dt_file, input_mode=args.input_mode, gt_mode=args.gt_mode)
        train_dataset = MotionDataset3D(args, args.subset_list, 'train')
        test_dataset = MotionDataset3D(args, args.subset_list, 'test')

    train_loader_3d = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)

    return train_loader_3d, test_loader, posetrack_loader_2d, instav_loader_2d, datareader