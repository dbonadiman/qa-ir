
import logging
import os.path
import json
import random
import torch
import numpy as np
import datetime


def timestamp_dir(basepath):
    return os.path.join(basepath, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

def merge_dict(dict_org, dict_alt):
    for key in dict_alt:
        if isinstance(dict_org[key], dict):
            dict_org[key] = merge_dict(dict_org[key], dict_alt[key])
        else:
            dict_org[key] = dict_alt[key]
    return dict_org

def assert_valid_config(file_name):
    if not os.path.isfile(file_name):
        raise RuntimeError(f'{file_name} is not a valid file')

def load_config(file_name, override):
    assert_valid_config(file_name)
    with open(file_name) as ifile:
        json_dict =  json.load(ifile)
        if override is not None:
            json_override = json.loads(override)
            return merge_dict(json_dict, json_override)
        else:
            return json_dict


def config_logger():
    import warnings
    warnings.simplefilter("ignore")
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.info('-'*50)

def set_seed(seed): 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def print_metrics(metrics):
    import tabulate
    return tabulate.tabulate(metrics.items(), headers=['Metric','Value'])
