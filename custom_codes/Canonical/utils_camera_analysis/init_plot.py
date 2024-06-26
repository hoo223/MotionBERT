from . import axes_3d, axes_2d, plt

def init_plot(self):
    self.plot3d.clear_output()
    self.plot2d.clear_output()
    
    if self.cam_mode == 'custom':
        with self.plot3d:
            fig = plt.figure(1, figsize=(15, 5),  layout='tight')
            fig.clear()
            self.ax_3d_1 = axes_3d(fig, loc=121, xlim=(-1,5), ylim=(-3,3), zlim=(0,2), view=(90,180), show_axis=True)   
            self.ax_3d_2 = axes_3d(fig, loc=122, xlim=(-1,5), ylim=(-3,3), zlim=(0,2), view=(0,90), show_axis=True)
            f = plt.show()    
            
        with self.plot2d:
            fig = plt.figure(2, figsize=(10, 5),  layout='tight') 
            fig.clear()
            self.ax_input = axes_2d(fig, loc=221, normalize=True, show_axis=False)
            self.ax_canonical = axes_2d(fig, loc=222, normalize=True, show_axis=False)
            self.ax_compare1 = axes_2d(fig, loc=223, normalize=True, show_axis=False)
            self.ax_compare2 = axes_2d(fig, loc=224, normalize=True, show_axis=False)
            f = plt.show() 
    elif self.cam_mode == 'h36m':
        with self.plot3d:
            self.fig_3d = plt.figure(1)
            self.fig_3d.clear()
            self.ax_3d_world = axes_3d(self.fig_3d, loc=121, xlim=(-4,4), ylim=(-5,5), zlim=(0,2), view=(90,180), show_axis=True)  
            self.ax_3d_cam = axes_3d(self.fig_3d, loc=122, xlim=(-1,1), ylim=(-1,1), zlim=(-1,1), view=(-90,-90), show_axis=True, title='root-centered cam_3d')
            f = plt.show()  
        with self.plot2d:
            self.fig_2d = plt.figure(2) 
            self.fig_2d.clear()
            self.ax_2ds = {}
            self.ax_2ds_canonical_3d_same_dist = {}
            self.ax_2ds_canonical_3d_same_z = {}
            self.ax_2ds_input_centering = {}
            
            if self.select_cam.value == 'total':
                for cam_name, loc in zip(self.cameras.keys(), [None, 221, 222, 223, 224]):
                    if cam_name == 'custom': continue
                    self.ax_2ds[cam_name] = axes_2d(self.fig_2d, loc=loc, normalize=True, show_axis=False, title=cam_name)
            else:
                self.ax_2ds[self.select_cam.value] = axes_2d(self.fig_2d, loc=221, normalize=True, show_axis=False, title=self.select_cam.value)
                self.ax_2ds_canonical_3d_same_dist[self.select_cam.value] = axes_2d(self.fig_2d, loc=222, normalize=True, show_axis=False, title=self.select_cam.value+'_canonical_3d_same_dist')
                self.ax_2ds_input_centering[self.select_cam.value] = axes_2d(self.fig_2d, loc=223, normalize=True, show_axis=False, title=self.select_cam.value+'_input_centering')
                self.ax_2ds_canonical_3d_same_z[self.select_cam.value] = axes_2d(self.fig_2d, loc=224, normalize=True, show_axis=False, title=self.select_cam.value+'_canonical_3d_same_z')
                
            f = plt.show()