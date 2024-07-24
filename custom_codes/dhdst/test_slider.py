# import sys
# sys.path.append('/home/hrai/codes/PoseAdaptor/')
# from lib_import import *
# from my_utils import *

# h36m_3d_world, h36m_cam_param = load_h36m()

# subject = 'S1'
# action = 'Walking'
# action_list = h36m_3d_world[subject].keys()
# pose3d_s1_walking = h36m_3d_world[subject][action]['positions'] # 3d skeleton sequence wrt world CS
# cam_info_s1_walking = h36m_3d_world[subject][action]['cameras']
# cam_param_s1_walking = get_cam_param(cam_info_s1_walking, subject, h36m_cam_param)

# # camera parameters
# W, H = cam_param_s1_walking['54138969']['W'], cam_param_s1_walking['54138969']['H']
# cam_54138969_ext = cam_param_s1_walking['54138969']['ext']
# cam_54138969_int = cam_param_s1_walking['54138969']['int']
# cam_54138969_proj = cam_param_s1_walking['54138969']['proj']
# cam_54138969_origin = cam_param_s1_walking['54138969']['C']

# # 3d trajectory
# torso_trajectory = get_part_traj(pose3d_s1_walking, 'torso')
# pelvis_trajectory = get_part_traj(pose3d_s1_walking, 'pelvis')
# l_hip_trajectory = get_part_traj(pose3d_s1_walking, 'l_hip')
# l_shoulder_trajectory = get_part_traj(pose3d_s1_walking, 'l_shoulder')
# r_shoulder_trajectory = get_part_traj(pose3d_s1_walking, 'r_shoulder')
# r_hip_trajectory = get_part_traj(pose3d_s1_walking, 'r_hip')
# lower_line_trajectory = get_part_traj(pose3d_s1_walking, 'lower_line')

# # World frame
# world_frame = generate_world_frame()


# from matplotlib.widgets import Slider, Button

# ref_pose = pose3d_s1_walking[0]
# pelvis_point = ref_pose[0]

# fig = plt.figure(3)
# fig.clear()
# ax = axes_3d(fig, loc=121, xlim=(-lim, lim), ylim=(-lim, lim), zlim=(0, 2), view=(20, -90))
# ax_2d = axes_2d(fig, loc=122, W=W, H=H)

# ax_azim  = fig.add_axes([0.1, 0.16, 0.3, 0.1]) # left, bottom, width, height
# ax_elev  = fig.add_axes([0.1, 0.08, 0.3, 0.1])
# ax_dist  = fig.add_axes([0.1, 0.0, 0.3, 0.1])
# ax_reset = fig.add_axes([0.8, 0.05, 0.1, 0.05])

# s_azim = Slider(ax = ax_azim, label = 'azimuth', valmin = -180, valmax = 180, valinit = 0, orientation="horizontal")
# s_elev = Slider(ax = ax_elev, label = 'elevation', valmin = 0, valmax = 30, valinit = 0, orientation="horizontal")
# s_dist = Slider(ax = ax_dist, label = 'distance', valmin = 1, valmax = 5, valinit = 3, orientation="horizontal")
# button = Button(ax_reset, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

# camera = Camera(origin=cam_origin, 
#                 calib_mat=calib_mat, 
#                 cam_default_R=cam_default_R, 
#                 roll=0,
#                 pitch=-elev,
#                 yaw=azim,
#                 IMAGE_HEIGHT=H, 
#                 IMAGE_WIDTH=W)

# def reset(event):
#     s_azim.reset()
#     s_elev.reset()
#     s_dist.reset()

# def update(val):
#     # update joint angles
#     azim = s_azim.val
#     elev = s_elev.val
#     dist = s_dist.val
    
#     s1 = time.time()
#     cam_origin = dist * azim_elev_to_vec(azim, elev, degrees=True) + pelvis_point
#     t1 = time.time() - s1
    
#     s2 = time.time()
#     camera.update_camera_parameter(origin=cam_origin, pitch=-elev, yaw=azim)
#     t2 = time.time() - s2

#     # 2d projection
#     ref_pose_projected = projection(ref_pose, camera.cam_proj)
    
#     # clear plot
#     t3 = time.time()
#     last_azim, last_elev = ax.azim, ax.elev
#     ax.cla()
#     ax.set_xlim(-lim, lim)
#     ax.set_ylim(-lim, lim)
#     ax.set_zlim(0, 2)
#     ax.view_init(elev=last_elev, azim=last_azim)
#     ax.set_aspect('equal', 'box')
#     # ax = axes_3d(fig, loc=121, xlim=(-lim, lim), ylim=(-lim, lim), zlim=(0, 2), view=(20, -90))
#     # ax_2d.cla()
#     # ax_2d = axes_2d(fig, loc=122, W=W, H=H)
    
#     # 3d plot
#     plt.sca(ax)
#     #ax = axes_3d(fig, loc=121, xlim=(-lim, lim), ylim=(-lim, lim), zlim=(0, 2), view=(20, -90))
#     draw_3d_pose(ax, ref_pose)
#     camera.cam_frame.draw3d()
    
#     # 2d plot
#     plt.sca(ax_2d)  
#     img = get_2d_pose_image(ref_pose_projected)
#     ax_2d.imshow(img)
#     t3 = time.time() - t3
#     plt.suptitle('t1: {:.5f}, elev: {:.5f}, dist: {:.5f}'.format(t1, t2, t3))
    

# # map the update function to all sliders
# button.on_clicked(reset)
# s_azim.on_changed(update)
# s_elev.on_changed(update)
# s_dist.on_changed(update)

# # initialize plot
# update(0)
# plt.show()