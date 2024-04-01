import ipywidgets as widgets

def get_inference_button():
    return widgets.Button(
        description='Inference',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='play'
    )
def get_visualize_button():
    return widgets.ToggleButton(
        value=False,
        description='Visualize',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Description',
        icon='check'
    )
    
def get_root_rel_button():
    return widgets.ToggleButton(
        value=False,
        description='Root Rel',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Description',
        icon='check'
    )
    
def get_analysis_button():
    return widgets.ToggleButton(
        value=False,
        description='Analysis',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Description',
        icon='check'
    )

def get_procrustes_button():
    return widgets.ToggleButton(
        value=False,
        description='Procrutes',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Description',
        icon='check'
    )

def get_reset_button(description):
    return widgets.Button(
        description=description,
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='refresh'
    )

def get_go_to_max_frame_button():
    return widgets.Button(
        description='Go to max frame',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='check' # (FontAwesome names without the `fa-` prefix)
    )