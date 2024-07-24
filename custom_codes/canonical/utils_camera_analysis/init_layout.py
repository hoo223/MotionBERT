from . import widgets, box_layout

def init_layout(self):
    if self.verbose:
        print('layout')
    if self.cam_mode == 'custom': self.init_custom_layout()
    elif self.cam_mode == 'h36m': self.init_h36m_layout()
    
def init_custom_layout(self):
    
    ui_select = widgets.HBox([self.select_subject, self.select_action])
    
    block_trans_x = widgets.HBox([self.trans_x_slider, self.trans_x_reset_button])
    block_trans_y = widgets.HBox([self.trans_y_slider, self.trans_y_reset_button])
    block_trans_z = widgets.HBox([self.trans_z_slider, self.trans_z_reset_button])
    block_rot_z = widgets.HBox([self.rot_z_slider, self.rot_z_reset_button])
    block_cam_roll = widgets.HBox([self.cam_roll_slider, self.cam_roll_reset_button])
    block_cam_pitch = widgets.HBox([self.cam_pitch_slider, self.cam_pitch_reset_button])
    block_cam_yaw = widgets.HBox([self.cam_yaw_slider, self.cam_yaw_reset_button])
    block_cam_height = widgets.HBox([self.cam_height_slider, self.cam_height_reset_button])
    
    ui_control_frame = widgets.HBox([self.frame_slider, self.frame_reset_button, self.total_reset_button])
    ui_control_pose = widgets.VBox([block_trans_x, block_trans_y, block_trans_z, block_rot_z])
    ui_control_cam = widgets.VBox([block_cam_roll, block_cam_pitch, block_cam_yaw, block_cam_height])
    ui_control = widgets.HBox([ui_control_pose, ui_control_cam])
    
    ui_monitor = widgets.VBox([self.vec_cam_origin_to_pelvis, self.dist_cam_origin_to_pelvis, self.text_vec_to_lhip, self.text_vec_to_rhip, self.text_dist_to_lhip, self.text_dist_to_rhip, self.text_total_length, self.text_l_ratio, self.text_r_ratio])
    
    # Integration
    ui_layer1 = widgets.HBox([self.plot3d, self.plot2d], layout=box_layout)
    ui_layer2 = widgets.HBox([ui_control_frame, self.set_compare1_button, self.set_compare2_button, self.set_line_mode_button])
    ui_layer3 = widgets.HBox([ui_select, ui_control, ui_monitor], layout=box_layout)
    self.ui = widgets.VBox([ui_layer1, ui_layer2, ui_layer3, self.test_out])
    
def init_h36m_layout(self):
    ui_select = widgets.HBox([self.select_subject, self.select_action, self.select_cam])
    ui_control_frame = widgets.HBox([self.frame_slider])
    
    # Integration
    ui_layer1 = widgets.HBox([self.plot3d, self.plot2d], layout=box_layout)
    ui_layer2 = widgets.HBox([ui_control_frame])
    ui_layer3 = widgets.HBox([ui_select], layout=box_layout)
    self.ui = widgets.VBox([self.test_out, ui_layer1, ui_layer2, ui_layer3])