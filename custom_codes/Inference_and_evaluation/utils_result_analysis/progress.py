import ipywidgets as widgets

def get_inference_progress():
    return widgets.FloatProgress(
        value=0,
        min=0,
        max=100,
        step=0.1,
        description='Testing:',
        bar_style='info',
        orientation='horizontal'
    )