import ipywidgets as widgets

def get_inputs_all_shape_text():
    return widgets.Text(
        value='',
        placeholder='no data',
        description='inputs_all:',
        disabled=True
    )
    
def get_gts_all_shape_text():
    return widgets.Text(
        value='',
        placeholder='no data',
        description='gts_all:',
        disabled=True
    )
    
def get_results_all_shape_text():
    return widgets.Text(
        value='',
        placeholder='no data',
        description='results_all:',
        disabled=True
    )

def get_batch_num_text():
    return widgets.BoundedIntText(
        value=0,
        min=0,
        max=0,
        step=1,
        description='Batch num:',
        disabled=False
    )
    
def get_frame_num_text():
    return widgets.BoundedIntText(
        value=1,
        min=1,
        max=1,
        step=1,
        description='Frame:',
        disabled=False
    )
    
def get_error_max_frame_text():
    return widgets.Text(
        value='',
        placeholder='no data',
        description='max error frame:',
        disabled=True
    )
    
def get_float_text(description):
    return widgets.FloatText(
        value=0.0,
        description=description,
        disabled=True
    )
    
def get_str_text(description):
    return widgets.Text(
        value='',
        placeholder='no data',
        description=description,
        disabled=True
    )