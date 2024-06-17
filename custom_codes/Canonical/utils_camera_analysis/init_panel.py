from . import get_list_select, get_str_text, get_frame_slider, get_trans_slider, get_button, widgets

def init_panel(self):
    if self.verbose:
        print('init panel')
    # select
    self.select_subject = get_list_select(self.subject_list, description='', rows=7)
    self.select_subject.layout.width = 'max-content'
    self.select_action = get_list_select(self.action_list, description='', rows=7)
    self.select_action.layout.width = 'max-content'
    self.select_cam = get_list_select(['total']+self.cam_list, description='', rows=5)
    self.select_cam.layout.width = 'max-content'
    
    # text
    self.vec_cam_origin_to_pelvis = get_str_text('vec_cam2pelvis')
    self.dist_cam_origin_to_pelvis = get_str_text('dist_cam2pelvis')
    self.text_vec_to_lhip = get_str_text('vec_to_lhip')
    self.text_vec_to_rhip = get_str_text('vec_to_rhip')
    self.text_dist_to_lhip = get_str_text('dist_to_lhip')
    self.text_dist_to_rhip = get_str_text('dist_to_rhip')
    self.text_total_length = get_str_text('total_length')
    self.text_l_ratio = get_str_text('l_ratio')
    self.text_r_ratio = get_str_text('r_ratio')
    
    # slider
    self.frame_slider = get_frame_slider(1)
    self.trans_x_slider = get_trans_slider(min=-2.0, max=2.0, description='trans x')
    self.trans_y_slider = get_trans_slider(min=-2.0, max=2.0, description='trans y')
    self.trans_z_slider = get_trans_slider(min=-2.0, max=2.0, description='trans z')
    self.rot_z_slider = get_trans_slider(min=-90, max=90, description='rot z')
    self.cam_roll_slider = get_trans_slider(min=-45, max=45, description='cam roll')
    self.cam_pitch_slider = get_trans_slider(min=-45, max=45, description='cam pitch')
    self.cam_yaw_slider = get_trans_slider(min=-45, max=45, description='cam yaw')
    self.cam_height_slider = get_trans_slider(value=self.cameras['custom'].origin[2], min=0.0, max=2.0, description='cam height')
    #widgets.jslink((self.frame_slider, 'value'), (self.frame_text, 'value'))
    
    # button
    self.frame_reset_button      = get_button('reset')
    self.total_reset_button      = get_button('total reset')
    self.trans_x_reset_button    = get_button('reset')
    self.trans_y_reset_button    = get_button('reset')
    self.trans_z_reset_button    = get_button('reset')
    self.rot_z_reset_button      = get_button('reset')
    self.cam_roll_reset_button   = get_button('reset')
    self.cam_pitch_reset_button  = get_button('reset')
    self.cam_yaw_reset_button    = get_button('reset')
    self.cam_height_reset_button = get_button('reset')
    self.set_compare1_button     = get_button('set compare1')
    self.set_compare2_button     = get_button('set compare2')
    self.set_line_mode_button    = get_button('set line mode')
    
    # plot         
    self.plot3d = widgets.Output()
    self.plot2d = widgets.Output()