import ipywidgets as widgets

def get_azim_slider():
    return widgets.IntSlider(
        value=-90,
        min=-180,
        max=180,
        step=1,
        description='azim:',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )

def get_elev_slider():
    return widgets.IntSlider(
        value=0,
        min=-180,
        max=180,
        step=1,
        description='elev:',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )

def get_zoom_slider():
    return widgets.FloatSlider(
        value=5.0,
        min=0.1,
        max=10,
        step=0.1,
        description='zoom:',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='f'
    )

def get_delay_slider():
    return widgets.FloatSlider(
        value=33.33,
        min=1,
        max=120,
        step=0.01,
        description='delay (ms):',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )

def get_frame_slider(max_frame):
    return widgets.IntSlider(
        value=1,
        min=1,
        max=max_frame,
        step=1,
        description='Frame:',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    
def get_trans_slider(value=0.0, min=-1.0, max=1.0, step=0.01, description='trans'):
    return widgets.FloatSlider(
        value=value,
        min=min,
        max=max,
        step=step,
        description=description,
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='.2f'
    )