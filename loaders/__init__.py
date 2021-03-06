import json

from loaders.hpa_loader import hpaLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'HPA': hpaLoader,
    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']
