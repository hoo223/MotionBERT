def set_line_mode(self, b):
    self.line_mode = not self.line_mode
    self.visualize_data()
    
def reset_frame(self, b):
    self.frame_slider.value = 0
    
def reset_total(self, b):
    self.trans_x_slider.value = 0
    self.trans_y_slider.value = 0
    self.trans_z_slider.value = 0
    self.rot_z_slider.value = 0
    self.cam_roll_slider.value = 0
    self.cam_pitch_slider.value = 0
    self.cam_yaw_slider.value = 0
    self.cam_height_slider.value = 1.0
    self.compare1_pose = None
    self.compare2_pose = None
    
    self.visualize_data()
    
def reset_trans_x(self, b):
    self.trans_x_slider.value = 0
    
def reset_trans_y(self, b):
    self.trans_y_slider.value = 0
    
def reset_trans_z(self, b):
    self.trans_z_slider.value = 0
    
def reset_rot_z(self, b):
    self.rot_z_slider.value = 0
    
def reset_cam_roll(self, b):
    self.cam_roll_slider.value = 0
    
def reset_cam_pitch(self, b):
    self.cam_pitch_slider.value = 0
    
def reset_cam_yaw(self, b):
    self.cam_yaw_slider.value = 0
    
def reset_cam_height(self, b):
    self.cam_height_slider.value = 1.0
    
def set_compare1(self, b):
    self.compare1_pose = self.pose_2d_norm_canonical.copy()
    self.visualize_data()
    
def set_compare2(self, b):
    self.compare2_pose = self.pose_2d_norm_canonical.copy()
    self.visualize_data()