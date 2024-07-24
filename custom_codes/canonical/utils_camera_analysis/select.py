from . import widgets

def get_list_select(_list, description, rows=15):
    return widgets.Select(
        options=_list,
        value=_list[0],
        rows=rows,
        description=description,
        disabled=False
    )