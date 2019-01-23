import os
import yaml
from .. import models

config_dir = models.configs.config_dir
_config_dir = models._configs._config_dir


def parse(config_path):
    """Parse a config file for cw-nets.

    Arguments
    ---------
    model_name : str
        Name of the model. This will be converted to paths to the YAML config
        files to parse.

    Returns
    -------
    config : dict
        A `dict` containing the information from the model config files.

    """

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        f.close()
    model_name = config['model_name']
    _config_path = os.path.join(_config_dir, model_name + '.yml')
    with open(_config_path, 'r') as f:
        _config = yaml.safe_load(f)
        f.close()
    # merge the elements of _config into config
    for k, v in _config:
        if k not in config:
            config[k] = v
    if config['model_name'] not in models.models:
        raise ValueError('{} is not a valid model name.'.format(
            config['model_name']))
    if not config['train'] and not config['infer']:
        raise ValueError('"train", "infer", or both must be true.')
    if config['train'] and config['data']['train_im_src_dirs'] is None:
        raise ValueError('"train_im_src_dirs" must be provided if training.')
    if config['train'] and config['data']['train_label_src_dirs'] is None:
        raise ValueError(
            '"train_label_src_dirs" must be provided if training.')
    if config['infer'] and config['data']['infer_im_src_dirs'] is None:
        raise ValueError('"infer_im_src_dirs" must be provided if "infer".')
    # TODO: IMPLEMENT UPDATING VALUES BASED ON EMPTY ELEMENTS HERE!

    return config
