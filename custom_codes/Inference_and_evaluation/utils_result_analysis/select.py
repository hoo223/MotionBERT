import ipywidgets as widgets


def get_subject_select(subject_list):
    return widgets.Select(
        options=subject_list,
        value=subject_list[0],
        rows=15,
        description='Subject:',
        disabled=False
    )

def get_action_select(action_list):
    return widgets.Select(
        options=action_list,
        value=action_list[0],
        rows=15,
        description='Action:',
        disabled=False,
    )

def get_cam_select(cam_list):
    return widgets.Select(
        options=cam_list,
        value=cam_list[0],
        rows=15,
        description='Cam:',
        disabled=False,
    )

def get_batch_select(batch_list):
    return widgets.Select(
        options=batch_list,
        value=batch_list[0],
        rows=15,
        description='Batch:',
        disabled=False,
    )

def get_part_select(part_list):
    return widgets.SelectMultiple(
        options=part_list,
        value=[part_list[0]],
        rows=len(part_list),
        description='Part',
        disabled=False
    )

def get_part_select2(part_list):
    return widgets.Select(
        options=part_list,
        value=part_list[0],
        rows=len(part_list),
        description='Part',
        disabled=False
    )