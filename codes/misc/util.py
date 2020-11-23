import os
import shutil

import numpy as np
import torch
import random
from distutils.dir_util import copy_tree
from shutil import rmtree
from torch.utils.tensorboard import SummaryWriter
import pickle


class writer(object):
    def __init__(self, config):
        self.summary_writer = SummaryWriter(log_dir=os.path.join(config.PATH.RUN.SAVE, 'tensorboard'))
        self.global_iter = 0

    def write_iter(self, **kwargs):
        self.global_iter+=1
        for key in kwargs:
            if key.endswith('loss'):
                self.summary_writer.add_scalar('Loss/train/{}'.format(key), kwargs[key], self.global_iter)

    def write_epoch(self, train_results, val_results, epoch):
        for split, results in [('train', train_results), ('val', val_results)]:
            for key in results:
                self.summary_writer.add_scalar('Metric/{}/{}/'.format(key, split), results[key], epoch)

def argmax_heatmap(heatmap):
    b, c, row, column = heatmap.shape
    heatmap = heatmap.reshape(b, c, -1)
    max_indices = heatmap.argmax(-1).data.cpu()
    keypoint = torch.zeros(b, c, 2)
    keypoint[:, :, 0] = torch.floor_divide(max_indices, column)
    keypoint[:, :, 1] = max_indices % column
    return keypoint


def get_config_key(config, before_keys=None):
    if before_keys is None:
        before_keys = []
    keys = []

    items = config.__dict__.items() if 'namespace' in str(config.__class__).lower() else config.items()
    for key, value in items:
        if 'attrdict' in str(value.__class__).lower():
            keys += get_config_key(value, before_keys + [key])
        else:
            keys.append(before_keys + [key])

    return keys



def save_config(config, save_path = None):
    if save_path is None:
        save_path = config.PATH.CONFIG
    if not config.debug:
        with open(save_path, 'wb') as f:
            pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)
    return

def get_attr_recursively(attrdict, keys, idx=0):
    key = keys[idx]
    # check key exists
    if key in attrdict:

        if 'attrdict' in str(attrdict[key].__class__).lower():
            value = get_attr_recursively(attrdict[key], keys, idx+1)
        else:
            value = attrdict[key]
    else:
        value = None
    return value

def set_attr_recursively(attrdict, keys, value, idx=0):
    if keys[idx] in attrdict:
        set_attr_recursively(attrdict[keys[idx]], keys, value, idx+1)
    elif len(keys)== idx+1:
        attrdict.__setattr__(keys[idx], value)
    else:
        attrdict.__setattr__(keys[idx], AttrDict())
        set_attr_recursively(attrdict[keys[idx]], keys, value, idx+1)
    return attrdict

def load_config(config, load_path=None):
    if load_path is None:
        load_path = config.PATH.CONFIG
    with open(load_path, 'rb') as f:
            loaded_config = pickle.load(f)

    current_config = convert_namespace_to_attrdict(config)
    loaded_config = convert_namespace_to_attrdict(loaded_config)

    current_config_key = get_config_key(current_config)
    loaded_config_key = get_config_key(loaded_config)

    new_config = AttrDict()
    for keys in loaded_config_key:
        loaded_value = get_attr_recursively(loaded_config, keys)
        new_config = set_attr_recursively(new_config, keys, loaded_value)
        if keys not in current_config_key:
            log(config, '[{}] does not exist in current config - value: [{}]'.format('.'.join(keys), loaded_value))

    for keys in current_config_key:
        if keys not in loaded_config_key:
            current_value = get_attr_recursively(current_config, keys)
            if current_config.yes_to_all:
                yes_or_no = 'yes'
            else:
                yes_or_no = input('Attribute [{}] not in loaded config. You want to use current one? (yes - continue / no - exit code)'.format('.'.join(keys))).lower()
            if yes_or_no == 'yes' or yes_or_no == 'y':
                log(config,'Missing Attribute {} = {}'.format('.'.join(keys), current_value))
                new_config = set_attr_recursively(new_config, keys, current_value)
            else:
                log(config,'Error : The old config [{}] is not suitable for this code'.format(config.PATH.CONFIG))
                raise

    new_config.debug = config.debug
    new_config.version = config.version
    new_config.PATH = config.PATH
    new_config.gpu = config.gpu
    new_config.mode = config.mode
    new_config.image_size = config.image_size

    if new_config.loss_weights != config.loss_weights:
        if current_config.yes_to_all:
            yes_or_no = 'yes'
        else:
            yes_or_no = input(
                'Attribute new loss_weight ? (yes - {} / no - {})'.format(config.loss_weights , new_config.loss_weights ))
        if yes_or_no == 'yes' or yes_or_no == 'y':
            new_config.loss_weights = config.loss_weights
            print(new_config.loss_weights)
            pass
        else:
            print(new_config.loss_weights)

    if new_config.decision_metric != config.decision_metric:
        if current_config.yes_to_all:
            yes_or_no = 'yes'
        else:
            yes_or_no = input(
                'Attribute new loss_weight ? (yes - {} / no - {})'.format(config.decision_metric , new_config.decision_metric ))
        if yes_or_no == 'yes' or yes_or_no == 'y':
            new_config.decision_metric = config.decision_metric
            print(new_config.decision_metric)
            pass
        else:
            print(new_config.decision_metric)


    new_config.batch_size = config.batch_size
    if config.model == 'unet3':
        new_config.model = config.model
    return new_config

def log(config, sentence):
    if not config.debug:
        log_path = os.path.join(config.PATH.RUN.SAVE, '{}log.txt'.format(config.log_other_file))
        with open(log_path,'a+') as f:
            f.write('{}\n'.format(sentence))
    print(sentence)
    return

def log_config(config, num_tab=0, value=None):
    if value is None:
        value = config

    if num_tab == 0:
        tab = ''
    else:
        tab = '-'*10 * num_tab
    items = value.__dict__.items() if 'namespace' in str(value.__class__).lower() else value.items()
    for key, value in items:
        if isinstance(value, AttrDict):
            sentence = ' {} {}'.format(tab, key)
            log(config, sentence)
            log_config(config, num_tab+1, value)
        else:
            sentence = '{} {:30.30} | {}'.format(tab, key, value)
            log(config, sentence)
    return


def save_codes(config):
    src = config.PATH.RUN.CODE
    des = os.path.join(config.PATH.RUN.SAVE, 'codes/')
    copy_tree(src,des)
    return

def set_gpu(config):
    os.environ['CUDA_VISIBLE_DEVICES']= config.gpu
    return

def set_seed(config):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    return



def assert_config(config):
    assert config.version is not None

    if not config.debug and (config.mode.train or config.train_val_test_split is not None):
        if os.path.exists(config.PATH.RUN.SAVE):
            while True:
                if config.version == 'debug':
                    yes_or_no = 'yes'
                else:
                    yes_or_no = input('Version {} already exists.. Do you want to overwrite this folder? (yes/no) '.format(
                                            config.version))
                if yes_or_no.lower() == 'yes' or yes_or_no.lower() == 'y':
                    rmtree(config.PATH.RUN.SAVE)
                    break
                elif yes_or_no.lower() == 'no' or yes_or_no.lower() == 'n':
                    exit('Use other version name')
                else:
                    pass

        make_save_dir(config)
    elif not os.path.exists(config.PATH.RUN.SAVE):
        make_save_dir(config)

    # train / val / test 비율 sum = 1 (preprocess 할때만 쓰임)
    if config.train_val_test_split is not None:
        assert np.sum(config.train_val_test_split) == 1
    return

def move_save_dir(from_path, to_path):
    items = os.listdir(from_path)
    os.makedirs(to_path)
    for item in items:
        src = os.path.join(from_path, item)
        des = os.path.join(to_path, item)
        shutil.move(src, des)

def make_save_dir(config):
    os.makedirs(config.PATH.RUN.SAVE)
    return


def convert_namespace_to_attrdict(config):
    new_attrdict = AttrDict()
    items = config.__dict__.items() if 'namespace' in str(config.__class__).lower() else config.items()
    for key, value in items:
        new_attrdict.__setattr__(key, value)
    return new_attrdict

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self