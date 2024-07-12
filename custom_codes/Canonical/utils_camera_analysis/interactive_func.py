from . import natsorted, get_pose_seq_and_cam_param, rotate_torso_by_R, Rotation, np

def update_select(self, subject, action):
    try:
        if len(self.h36m_3d_world._data.keys()) > 0:
            # with self.test_out:
            #     print('update select')
            if self.subject != subject:
                #self.select_subject.options = natsorted(list(self.h36m_3d_world._data.keys()))
                self.select_subject.value = subject
                self.select_action.options = natsorted(list(self.h36m_3d_world._data[subject].keys()))
                self.select_action.value = self.select_action.options[0]
                self.subject = subject
                self.action = self.select_action.value
            elif self.action != action:
                self.select_action.value = action
                self.action = action
            self.pose_3d_list, self.cam_param = get_pose_seq_and_cam_param(self.h36m_3d_world, self.h36m_cam_param, subject, action)
            self.frame_slider.max = len(self.pose_3d_list)-1
            self.update_ref_pose()
            self.visualize_data()
    except Exception as e:
        with self.test_out:
            print(subject, action)
            print('select error', e)
            print()
            
def update_frame(self, frame):
    if self.frame_num != frame:
        self.frame_num = frame
        self.update_ref_pose()
        self.visualize_data()
        
def update_cam(self, cam_name):
    self.init_plot()
    self.visualize_data()
        
def update_cam_pose(self, roll, pitch, yaw, height):
    if self.verbose: print('update cam pose')
    self.cameras['custom'].update_camera_parameter(origin=np.array([self.cameras['custom'].origin[0], self.cameras['custom'].origin[1], height]), roll=roll, pitch=pitch, yaw=yaw)
    self.visualize_data()
        
def update_ref_pose(self):
    frame_num = self.frame_num
    ref_pose = self.pose_3d_list[frame_num].copy()
    if self.cam_mode == 'custom':
        ref_pose = rotate_torso_by_R(ref_pose, Rotation.from_rotvec([0, 0, -np.pi/2]).as_matrix())
        ref_pose[:, :2] -= ref_pose[0, :2]
    self.ref_pose = ref_pose
        
def update_pose_diff(self, trans_x, trans_y, trans_z, rot_z):
    self.dx = trans_x
    self.dy = trans_y
    self.dz = trans_z
    self.rz = rot_z
    self.visualize_data()