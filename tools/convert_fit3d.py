# import os
# import sys
# import pickle
# import numpy as np
# import random
# sys.path.insert(0, os.getcwd())
# from lib.utils.tools import read_pkl
# from lib.data.datareader_fit3d import DataReaderFIT3D
# from tqdm import tqdm


# def save_clips(subset_name, root_path, train_data, train_labels):
#     len_train = len(train_data)
#     print(len_train)
#     save_path = os.path.join(root_path, subset_name)
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     for i in tqdm(range(len_train)):
#         data_input, data_label = train_data[i], train_labels[i]
#         data_dict = {
#             "data_input": data_input,
#             "data_label": data_label
#         }
#         with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as myprofile:  
#             pickle.dump(data_dict, myprofile)
            
# datareader = DataReaderFIT3D(n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243, dt_file = 'fit3d_gt_all_test.pkl', dt_root='data/motion3d/', mode='joint3d_image')

# train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()
# #print(train_data.shape, test_data.shape)
# assert len(train_data) == len(train_labels)
# assert len(test_data) == len(test_labels)

# root_path = "data/motion3d/MB3D_f243s81/FIT3D-GT-ALL_TEST/"
# if not os.path.exists(root_path):
#     os.makedirs(root_path)

# save_clips("train", root_path, train_data, train_labels)
# save_clips("test", root_path, test_data, test_labels)



import os
import sys
import pickle
import numpy as np
import random
sys.path.insert(0, os.getcwd())
from lib.utils.tools import read_pkl
from lib.data.datareader_fit3d import DataReaderFIT3D
from tqdm import tqdm


def save_clips(subset_name, root_path, train_data, train_labels):
    len_train = len(train_data)
    print(len_train)
    save_path = os.path.join(root_path, subset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in tqdm(range(len_train)):
        data_input, data_label = train_data[i], train_labels[i]
        data_dict = {
            "data_input": data_input,
            "data_label": data_label
        }
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as myprofile:  
            pickle.dump(data_dict, myprofile)
            
#datareader = DataReaderFIT3D(n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243, dt_file = 'fit3d_gt_canonical_3d_same_z_all_test.pkl', dt_root='data/motion3d/', input_mode='joint_2d_from_canonical_3d', gt_mode='cam_3d')
datareader = DataReaderFIT3D(n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243, dt_file = 'fit3d_gt_all_test.pkl', dt_root='data/motion3d/', input_mode='joint_2d', gt_mode='joint3d_image')

train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()
#print(train_data.shape, test_data.shape)
assert len(train_data) == len(train_labels)
assert len(test_data) == len(test_labels)

#root_path = "data/motion3d/MB3D_f243s81/FIT3D-GT-CAM_NO_FACTOR-INPUT_FROM_3D_CANONICAL_SAME_DIST-ALL_TEST/"
root_path = "data/motion3d/MB3D_f243s81/FIT3D-GT-ALL_TEST/"
if not os.path.exists(root_path):
    os.makedirs(root_path)

save_clips("train", root_path, train_data, train_labels)
save_clips("test", root_path, test_data, test_labels)

