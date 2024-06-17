from . import widgets

def get_toggle_button(description):
    return widgets.ToggleButton(
        value=False,
        description=description,
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Description',
        icon='check'
    ) 

def get_button(description):
    return widgets.Button(
        description=description,
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='refresh'
    )