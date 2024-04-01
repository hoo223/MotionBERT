import ipywidgets as widgets

def get_play_vis_button():
    return widgets.Play(
        interval=33.33,
        value=1,
        min=1,
        max=1,
        step=1,
        description="Press play",
        disabled=False
    )