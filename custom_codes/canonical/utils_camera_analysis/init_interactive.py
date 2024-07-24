from . import widgets

def init_interactive(self):
    if self.verbose:
        print('interactive')
    if self.cam_mode == 'custom': 
        self.custom_interactive()
        self.custom_button_function()
    elif self.cam_mode == 'h36m': 
        self.h36m_interactive()
        self.h36m_button_function()

def custom_interactive(self):
    self.select_interact = widgets.interactive(self.update_select, subject=self.select_subject, action=self.select_action)
    self.select_frame_interact = widgets.interactive(self.update_frame, frame=self.frame_slider)
    self.trans_interact = widgets.interactive(self.update_pose_diff, trans_x=self.trans_x_slider, trans_y=self.trans_y_slider, trans_z=self.trans_z_slider, rot_z=self.rot_z_slider)
    self.cam_interact = widgets.interactive(self.update_cam_pose, roll=self.cam_roll_slider, pitch=self.cam_pitch_slider, yaw=self.cam_yaw_slider, height=self.cam_height_slider)

def h36m_interactive(self):
    self.select_subject_interact = widgets.interactive(self.update_select, subject=self.select_subject, action=self.select_action)
    self.select_frame_interact = widgets.interactive(self.update_frame, frame=self.frame_slider)
    self.select_cam_interact = widgets.interactive(self.update_cam, cam_name=self.select_cam)
    
def custom_button_function(self):
    # reset
    self.frame_reset_button.on_click(self.reset_frame)
    self.total_reset_button.on_click(self.reset_total)
    self.trans_x_reset_button.on_click(self.reset_trans_x)
    self.trans_y_reset_button.on_click(self.reset_trans_y)
    self.trans_z_reset_button.on_click(self.reset_trans_z)
    self.rot_z_reset_button.on_click(self.reset_rot_z)
    self.cam_roll_reset_button.on_click(self.reset_cam_roll)
    self.cam_pitch_reset_button.on_click(self.reset_cam_pitch)
    self.cam_yaw_reset_button.on_click(self.reset_cam_yaw)
    self.cam_height_reset_button.on_click(self.reset_cam_height)
    
    # compare
    self.set_compare1_button.on_click(self.set_compare1)
    self.set_compare2_button.on_click(self.set_compare2)
    
    # set mode
    self.set_line_mode_button.on_click(self.set_line_mode)
    
def h36m_button_function(self):
    # reset
    self.frame_reset_button.on_click(self.reset_frame)