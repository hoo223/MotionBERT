from . import rotate_torso_by_R, projection, normalize_screen_coordinates, draw_3d_pose, draw_2d_pose, clear_axes, get_rootrel_pose, World2CameraCoordinate, draw3d_arrow, rotation_matrix_from_vectors
from . import Rotation, radians, cos, sin, np, plt

def update_world_3d(self):
    if self.cam_mode == 'custom':
        if self.line_mode:
            self.world_3d = np.array([[0, 0, 0], [0, -0.2, 0], [0, 0.2, 0]]) + np.array([self.cameras['custom'].origin[0]+self.init_dx, self.init_dy, self.init_dz+self.cameras['custom'].origin[2]]) + np.array([self.dx, self.dy, self.dz])
        else:
            self.world_3d = self.ref_pose.copy() + np.array([self.cameras['custom'].origin[0]+self.init_dx, self.init_dy, self.init_dz]) + np.array([self.dx, self.dy, self.dz])
        self.world_3d = rotate_torso_by_R(self.world_3d, Rotation.from_rotvec([0, 0, radians(self.rz)]).as_matrix())
    elif self.cam_mode == 'h36m':
        self.world_3d = self.ref_pose.copy()
        
def update_canonical_3d(self, cam_name, mode='same_dist'):
    canonical_3d = self.world_3d.copy()
    cam_origin = self.cameras[cam_name].origin
    pelvis = self.world_3d[0]
    vec_cam_origin_to_pelvis = pelvis - cam_origin
    mag_cam_origin_to_pelvis = np.linalg.norm(vec_cam_origin_to_pelvis)
    vec_cam_forward = self.cameras[cam_name].R[2]
    if mode == 'same_dist':
        vec_to_new_pelvis = vec_cam_forward * mag_cam_origin_to_pelvis
    elif mode == 'same_z':
        vec_to_new_pelvis = vec_cam_forward * np.dot(vec_cam_origin_to_pelvis, vec_cam_forward)/np.dot(vec_cam_forward, vec_cam_forward)
    canonical_pelvis = cam_origin + vec_to_new_pelvis
    canonical_3d = canonical_3d - pelvis + canonical_pelvis
    return vec_cam_origin_to_pelvis, vec_cam_forward*mag_cam_origin_to_pelvis, canonical_3d

def update_cam_3d(self, cam_name):
    cam_ext = self.cameras[cam_name].ext_mat
    cam_3d = World2CameraCoordinate(self.world_3d.copy(), cam_ext)
    return cam_3d

def generate_2d_pose(self, pose_3d, cam_name):      
    cam_proj = self.cameras[cam_name].cam_proj
    cam_ext = self.cameras[cam_name].ext_mat
    W = self.cameras[cam_name].IMAGE_WIDTH
    H = self.cameras[cam_name].IMAGE_HEIGHT
    
    pose_2d = projection(pose_3d, cam_proj)[..., :2]
    pose_2d_norm = normalize_screen_coordinates(pose_2d, W, H)
    pose_2d_norm_centered = pose_2d_norm - pose_2d_norm[0]
    
    return pose_2d, pose_2d_norm, pose_2d_norm_centered
    
def calculate_error(self, cam_name):
    cam_origin = self.cameras[cam_name].origin
    pelvis = self.world_3d[0]
    vec_cam_origin_to_pelvis = pelvis - cam_origin
    mag = np.linalg.norm(vec_cam_origin_to_pelvis)
    cam_yaw = self.cam_yaw_slider.value
    
    self.dist_cam_origin_to_pelvis.value = str(mag)
    self.vec_cam_origin_to_pelvis.value = f"z: {mag * cos(radians(cam_yaw)):.2f} x: {-mag * sin(radians(cam_yaw)):.2f}"
    
    if self.line_mode:
        self.vec_to_lhip = self.pose_2d_norm_canonical[1] - self.pose_2d_norm_canonical[0]
        self.vec_to_rhip = self.pose_2d_norm_canonical[2] - self.pose_2d_norm_canonical[0]
        self.dist_to_lhip = np.linalg.norm(self.vec_to_lhip)
        self.dist_to_rhip = np.linalg.norm(self.vec_to_rhip)    
        self.total_length = self.dist_to_lhip + self.dist_to_rhip
        self.l_ratio = self.dist_to_lhip / self.total_length
        self.r_ratio = self.dist_to_rhip / self.total_length
        
        self.text_vec_to_lhip.value = f"x: {self.vec_to_lhip[0]:.2f} y: {self.vec_to_lhip[1]:.2f}"
        self.text_vec_to_rhip.value = f"x: {self.vec_to_rhip[0]:.2f} y: {self.vec_to_rhip[1]:.2f}"
        self.text_dist_to_lhip.value = f"{self.dist_to_lhip:.2f}"
        self.text_dist_to_rhip.value = f"{self.dist_to_rhip:.2f}"
        self.text_total_length.value = f"{self.total_length:.2f}"
        self.text_l_ratio.value = f"{self.l_ratio:.2f}"
        self.text_r_ratio.value = f"{self.r_ratio:.2f}"
    
def visualize_data(self):
    if self.cam_mode == 'custom':
        self.update_world_3d()
        self.cam_3d = self.update_cam_3d('custom')
        self.pose_2d, self.pose_2d_norm, self.pose_2d_norm_centered = self.generate_2d_pose(self.cam_3d, 'custom')
        if self.line_mode:   dataset_type = 'base'
        else:                dataset_type = 'h36m'
        self.calculate_error(cam_name='custom')
        
        with self.plot3d:
            clear_axes([self.ax_3d_1, self.ax_3d_2])
            plt.sca(self.ax_3d_1)
            self.cameras['custom'].cam_frame.draw3d()
            plt.sca(self.ax_3d_2)
            self.cameras['custom'].cam_frame.draw3d()
            draw_3d_pose(self.ax_3d_1, self.world_3d, dataset=dataset_type)
            draw_3d_pose(self.ax_3d_2, self.world_3d, dataset=dataset_type)
            
        with self.plot2d:
            clear_axes([self.ax_input, self.ax_canonical, self.ax_compare1, self.ax_compare2])
            draw_2d_pose(self.ax_input, self.pose_2d_norm_centered, normalize=True, dataset=dataset_type)
            #draw_2d_pose(self.ax_canonical, self.pose_2d_norm_canonical, normalize=True, dataset=dataset_type)
            if self.compare1_pose is not None:
                draw_2d_pose(self.ax_compare1, self.compare1_pose, normalize=True, dataset=dataset_type)
            if self.compare2_pose is not None:
                draw_2d_pose(self.ax_compare2, self.compare2_pose, normalize=True, dataset=dataset_type)
    elif self.cam_mode == 'h36m':
        self.update_world_3d()
        self.canonical_3ds_same_dist = {}
        self.canonical_3ds_same_z = {}
        self.vec_cam_origin_to_pelvises = {}
        self.vec_cam_forwards = {}
        self.cam_3ds = {}
        self.pose_2ds = {}
        self.pose_2d_norms = {}
        self.pose_2d_norm_canonicals = {}
        
        if self.select_cam.value == 'total': cam_list = self.cameras.keys()
        else:                                cam_list = [self.select_cam.value]
        
        # clear 3D axes
        with self.plot3d:
            clear_axes([self.ax_3d_world, self.ax_3d_cam])
        
        for cam_name in cam_list:
            if cam_name == 'custom': continue
            self.cam_3ds[cam_name] = self.update_cam_3d(cam_name)
            pose_2ds, pose_2d_norms, pose_2d_norms_centered = self.generate_2d_pose(self.world_3d, cam_name)
            
            if self.select_cam.value != 'total':
                self.vec_cam_origin_to_pelvises[cam_name], self.vec_cam_forwards[cam_name], self.canonical_3ds_same_dist[cam_name] = self.update_canonical_3d(cam_name, 'same_dist')
                _, _, self.canonical_3ds_same_z[cam_name] = self.update_canonical_3d(cam_name, 'same_z')
                pose_2d_canonical_3d_same_dist, pose_2d_norms_canonical_3d_same_dist, _ = self.generate_2d_pose(self.canonical_3ds_same_dist[cam_name], cam_name)
                pose_2d_canonical_3d_same_z, pose_2d_norms_canonical_3d_same_z, _ = self.generate_2d_pose(self.canonical_3ds_same_z[cam_name], cam_name)
                #with self.test_out:
                #    print(pose_2d_canonical_3d_same_dist[0], pose_2d_canonical_3d_same_z[0])

            with self.plot3d:
                plt.sca(self.ax_3d_world)
                self.cameras[cam_name].cam_frame.draw3d()
                draw_3d_pose(self.ax_3d_world, self.world_3d, dataset='h36m')
                if self.select_cam.value != 'total':
                    draw3d_arrow(arrow_location=self.cameras[cam_name].origin, 
                                arrow_vector=self.vec_cam_origin_to_pelvises[cam_name], 
                                head_length=0.5,
                                color='r',
                                ax=self.ax_3d_world)
                    draw3d_arrow(arrow_location=self.cameras[cam_name].origin, 
                                arrow_vector=self.vec_cam_forwards[cam_name], 
                                head_length=0.5,
                                color='b',
                                ax=self.ax_3d_world)
                    #draw_3d_pose(self.ax_3d_world, self.canonical_3ds_same_dist[cam_name], dataset='h36m', color='r')
                    draw_3d_pose(self.ax_3d_world, self.canonical_3ds_same_z[cam_name], dataset='h36m', color='b')
                    draw_3d_pose(self.ax_3d_cam, get_rootrel_pose(self.cam_3ds[cam_name]), dataset='h36m')
            with self.plot2d:
                if self.select_cam.value != 'total':
                    clear_axes([self.ax_2ds[cam_name], self.ax_2ds_canonical_3d_same_dist[cam_name], self.ax_2ds_canonical_3d_same_z[cam_name], self.ax_2ds_input_centering[cam_name]])
                    draw_2d_pose(self.ax_2ds_canonical_3d_same_dist[cam_name], pose_2d_norms_canonical_3d_same_dist, normalize=True, dataset='h36m')
                    draw_2d_pose(self.ax_2ds_canonical_3d_same_z[cam_name], pose_2d_norms_canonical_3d_same_z, normalize=True, dataset='h36m')
                    draw_2d_pose(self.ax_2ds_input_centering[cam_name], pose_2d_norms_centered, normalize=True, dataset='h36m')
                    #draw_2d_pose(self.ax_2ds_canonical_3d_same_z[cam_name], , normalize=True, dataset='h36m')
                clear_axes(self.ax_2ds[cam_name])
                draw_2d_pose(self.ax_2ds[cam_name], pose_2d_norms, normalize=True, dataset='h36m')
            
            