from . import load_h36m, get_pose_seq_and_cam_param, natsorted

def load_h36m_dataset(self):
    if self.verbose: print('load_h36m')
    self.h36m_3d_world, self.h36m_cam_param = load_h36m()
    self.subject_list = natsorted(list(self.h36m_3d_world._data.keys()))
    self.action_list = natsorted(list(self.h36m_3d_world._data[self.subject_list[0]].keys()))
    self.subject = self.subject_list[0]
    self.action = self.action_list[0]
    self.pose_3d_list, self.cam_param = get_pose_seq_and_cam_param(self.h36m_3d_world, self.h36m_cam_param, self.subject, self.action)
    self.cam_list = natsorted(list(self.cam_param.keys()))