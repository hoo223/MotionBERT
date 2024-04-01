import ipywidgets as widgets

def get_model_list_dropdown(options):
    return widgets.Dropdown(
        options=options,
        value=options[0],
        description='Model:',
        disabled=False,
    )


def get_dataset_list_dropdown(options):
    return widgets.Dropdown(
        options=options,
        value=options[0],
        description='Dataset:',
        disabled=False,
    )