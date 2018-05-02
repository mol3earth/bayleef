import os
import wget
import re

from datetime import datetime


def get_path(response, root, dataset):
    """
    """
    if isinstance(response, dict):
        response = [response]

    try:
        for data in response:
            scene_id = data['entityId']
            s = data['summary']
            date = datetime.strptime(data['acquisitionDate'], '%Y-%m-%d')
            pathstr, rowstr = re.findall(r'Path: \b\d+\b|Row: \b\d+\b', s)
            path = pathstr.split(' ')[1]
            row = rowstr.split(' ')[1]
            return os.path.join(root, dataset, str(date.year), row, path, scene_id)

    except Exception as e:
        print('Failed to process request: {}'.format(e))


def keys_to_lower(dictionary):
    """
    """
    for key in dictionary.keys():
        if isinstance(dictionary[key], dict):
            keys_to_lower(dictionary[key])
        dictionary[key.lower()] = dictionary.pop(key)


def apply_dict(dictionary, func, *args, **kwargs):
    """
    """
    for key in dictionary.keys():
        if isinstance(dictionary[key], dict):
            apply_dict(dictionary[key], func)
        dictionary[key] = func(dictionary[key], *args, **kwargs)
