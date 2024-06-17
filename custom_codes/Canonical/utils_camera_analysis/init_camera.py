from . import Camera, T_to_C, np

def init_camera(self):
    target = '54138969'
    calib_mat = self.cam_param[target]['int']['calibration_matrix']
    self.init_custom_camera(W=self.cam_param[target]['W'], H=self.cam_param[target]['H'],
                fx=calib_mat[0][0], fy=calib_mat[1][1], cx=calib_mat[0][2], cy=calib_mat[1][2])
    self.init_h36m_camera()
    
def init_h36m_camera(self):
    if self.verbose: print('init_h36m_camera')
    # camera parameter
    for cam_name in self.cam_param.keys():
        W, H = self.cam_param[cam_name]['W'], self.cam_param[cam_name]['H']
        calib_mat = self.cam_param[cam_name]['int']['calibration_matrix']
        R = np.array(self.cam_param[cam_name]['ext']['R'])
        t = np.array(self.cam_param[cam_name]['ext']['t'])/1000
        C = T_to_C(R, t)

        camera = Camera(origin=C, 
                        calib_mat=calib_mat, 
                        cam_default_R=R, 
                        IMAGE_HEIGHT=H, 
                        IMAGE_WIDTH=W,
                        cam_name=cam_name)
        self.cameras[cam_name] = camera
    
def init_custom_camera(self, W=1000, H=1000, cam_height=1.0, 
                fx=1.0, fy=1.0, cx=500.0, cy=500.0,
                init_roll_angle=0, init_pitch_angle=0, init_yaw_angle=0):
    if self.verbose: print('init_custom_camera')
    calib_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])    
            
    # camera parameter
    cam_origin = np.array([0, 0, cam_height])

    forward = [1, 0, 0]
    left = [0, -1, 0]
    up = np.cross(forward, left)
    cam_default_R = np.array([left, up, forward]) # default camera orientation

    roll_angle = init_roll_angle
    pitch_angle = init_pitch_angle
    yaw_angle = init_yaw_angle

    camera = Camera(origin=cam_origin, 
                    calib_mat=calib_mat, 
                    cam_default_R=cam_default_R, 
                    roll=roll_angle,
                    pitch=pitch_angle,
                    yaw=yaw_angle,
                    IMAGE_HEIGHT=H, 
                    IMAGE_WIDTH=W)
    self.cameras['custom'] = camera