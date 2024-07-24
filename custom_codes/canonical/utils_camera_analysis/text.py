from . import widgets
    
def get_str_text(description):
    return widgets.Text(
        value='',
        placeholder='no data',
        description=description,
        disabled=True
    )