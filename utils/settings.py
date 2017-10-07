import os
import yaml

from .containers import DefaultNoneAttrDict, Folder

VERSION = 32


def _config_walk(x, object_type=dict, dict_convert=lambda x: x, replace_type=DefaultNoneAttrDict):
    if isinstance(x, object_type):
        return replace_type((k.lower(), _config_walk(v, object_type=object_type, dict_convert=dict_convert, replace_type=replace_type)) for k, v in dict_convert(x).iteritems())
    elif isinstance(x, (tuple, list)):
        return [_config_walk(val, object_type=object_type, dict_convert=dict_convert, replace_type=replace_type) for val in x]
    else:
        return x


def get_config(path=None):
    source_root = Folder(__file__).up()
    default_config_locations = [source_root.join('config.yaml')]
    config_locations = [path] if path else default_config_locations
    for config_path in config_locations:
        if os.path.isfile(config_path):
            break
    else:
        return DefaultNoneAttrDict()
    with open(config_path) as f:
        config = yaml.load(f)
    config = _config_walk(config)
    config.paths = Folder(config.general.basedir or '.')
    if config.accounts:
        for provider_name in config.accounts.keys():
            active_accounts = [acc for acc in config.accounts[provider_name] if not acc.disabled]
            config.accounts[provider_name] = active_accounts
    return config


config = get_config()
paths = config.paths
