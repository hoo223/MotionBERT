from . import widgets

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