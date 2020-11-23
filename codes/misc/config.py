import argparse
import os
import torch
import numpy as np
from misc import util


def parse_args():
    parser = argparse.ArgumentParser(description='TMI experiments')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam,')
    parser.add_argument('--loss', type=str, default='CustomLoss2', help='customloss')
    parser.add_argument('--loss_weights', type=float, nargs='*', default=[1, 1, 1])
    parser.add_argument('--metrics', type=str, nargs='*', default=['mae', 'rmse', 'iou','weighted_sum', 'mre'], help='mae, rmse, iou ...')
    parser.add_argument('--decision_metric', type=str, default='mae_mm', help='loss, mae, rmse, ...')
    parser.add_argument('--scheduler', type=str, help='')
    parser.add_argument('--aug_p', type=float, nargs='*', default=None, help=' (0.3, 0.2, 0.3, 0.3, 0.3, 0.2)')
    parser.add_argument('--model', type=str, default='unet', help='maskrcnn, unet')

    parser.add_argument('--use_hint', action='store_true', default=False, help='If activated, dataloader generate random hint')
    parser.add_argument('--hint_num_dist_type', type=str, default='default', help='default, always_hint')
    parser.add_argument('--success_standard', type=float, default=[0.1,0.3,0.5,0.7,0.9,1,1.5,2,2.5,3,3.5],nargs='*', help='success / failure standard for test with hint')
    parser.add_argument('--enchint', action='store_true', default=False, help='enchint')
    parser.add_argument('--image_size', type=int, nargs='*', default=(400, 400), help='height(row), width(column)')
    parser.add_argument('--n_bbox', type=int, default=1)
    parser.add_argument('--n_class', type=int, default=13)
    parser.add_argument('--heatmap_std', type=float, default=2.5, help='2.5')
    parser.add_argument('--sampler', type=str)
    parser.add_argument('--train_val_test_split', type=float, nargs='*', default=None, help='[train, val, test], [0.5, 0.1, 0.4]')
    parser.add_argument('--tensorboard', action='store_true', default=False)
    parser.add_argument('--version', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_split_seed', type=int, default=42,
                        help='default = 42, Caution: this will change data split.')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--run_root', type=str, default='../')
    parser.add_argument('--config_path', type=str, help='load config')

    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--yes_to_all', action='store_true', default=False)
    parser.add_argument('--datasplit', type=str, default='tr0.5_dev0.1_t0.4',
                        help='tr0.1_dev0.1_t0.8, tr0.2_dev0.1_t0.7, tr0.5_dev0.1_t0.4')
    parser.add_argument('--description', type=str, help="explain what experiment it is")
    parser.add_argument('--data_root', type=str, default='/data/')

    parser.add_argument('--hint_std', type=float, default=7.5, help="")

    parser.add_argument('--opt_unit_dist_unit_vector_size', type=float, default=1)

    parser.add_argument('--log_other_file', type=str, default='')
    args = parser.parse_args()
    return args

def get_config():

    config = parse_args()
    config = set_static_variables(config)

    # RUN MODE
    config.mode = util.AttrDict()
    config.mode.train = config.train
    config.mode.val = config.val
    config.mode.test = config.test

    config.opt = set_opt_variables(config)
    config.hint = set_hint_variables(config)
    config.aug = set_transform(config)

    config.PATH = set_config_path(config)

    if not config.mode.train:
        config = util.load_config(config)
        config.LOADPATH = config.PATH.RUN.SAVE

    if config.debug:
        config = update_version(config, 'debug')
        config.epochs = 2
        config.num_workers = 0



    if not config.mode.train: # val / test
        config = update_version(config, os.path.join(config.version,
                                        'test'))


    config = set_static_variables(config)
    config.PATH = set_config_path(config)

    util.assert_config(config)
    util.log_config(config)
    util.save_config(config)
    util.save_codes(config)

    util.set_seed(config)
    util.set_gpu(config)

    return config


def set_transform(config):
    aug = util.AttrDict()
    aug.p = config.aug_p

    return aug
def update_version(config, version):
    config.version = version
    config.PATH = set_config_path(config)
    return config


def set_hint_variables(config):
    hint = util.AttrDict()
    hint.channel = config.numkey
    hint.enc_dist = np.array([1/13 for i in range(13)])
    hint.num_dist = [1 / 8, 1 / 2, 1 / 4, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512, 1 / 1024,
                                1 / 2048, 1 / 4096, 1 / 4096]
    return hint

def set_opt_variables(config):
    opt = util.AttrDict()
    opt.flag = True
    opt.opt_unit_dist = True
    opt.pairs = [
        [(0, 1), (5, 8), (5, 6), (10, 11), (0, 3), (1, 2), (6, 7), (1, 3), (11, 12), (6, 8), (6, 9), (7, 9),(1, 4), (2, 4)], [(7, 8), (2, 3), (5, 6), (0, 1), (10, 11), (10, 12), (0, 2), (5, 7)]]

    return opt


def set_config_path(config):
    # PATH
    PATH = util.AttrDict()
    PATH.RUN = util.AttrDict()
    PATH.RUN.ROOT = config.run_root
    PATH.RUN.CODE = os.path.join('../', 'codes') #only used when save code files
    PATH.RUN.SAVE = os.path.join(PATH.RUN.ROOT, 'save/{}/'.format(config.version))
    PATH.RUN.SAVE_FIG = os.path.join(PATH.RUN.SAVE, 'result_images_forward')
    PATH.RUN.SAVE_HINT_RESULT = os.path.join(PATH.RUN.SAVE, 'numpy_items')

    PATH.DATA = util.AttrDict()
    PATH.DATA.ROOT = config.data_root
    PATH.DATA.VERSION = os.path.join(PATH.DATA.ROOT, 'data', 'data_200820/')
    PATH.DATA.TABLE = os.path.join(PATH.DATA.VERSION, config.datasplit + '_6504')
    PATH.DATA.TRAIN = os.path.join(PATH.DATA.TABLE , 'train.json')
    PATH.DATA.VAL = os.path.join(PATH.DATA.TABLE , 'val.json')
    PATH.DATA.TEST = os.path.join(PATH.DATA.TABLE , 'test.json')

    PATH.CONFIG = os.path.join(PATH.RUN.SAVE, 'config.pickle')
    return PATH


def set_static_variables(config):

    config.DICT_KEY = util.AttrDict()

    config.DICT_KEY.IMAGE = 'image'
    config.DICT_KEY.BBOX = 'bbox_minX_minY_clip'  # 'bbox_minX_maxY'
    config.DICT_KEY.POINTS = 'spine_x_y'
    config.numkey = 13
    config.n_class = config.numkey
    config.DICT_KEY.RAW_SIZE = 'raw_size_row_col'
    config.DICT_KEY.PSPACE = 'pixelSpacing'

    config.criterion = util.AttrDict()
    config.criterion.minimize = ['loss', 'mae_mm', 'rmse_mm', 'mae_pixel', 'rmse_pixel', 'mre_mm', 'mre_mm_weighted_sum', 'mae_mm_weighted_sum', 'mae_pixel_weighted_sum']
    config.criterion.maximize = ['iou', 'improved_mae_mm', 'improved_mae_pixel','improved_rmse_mm', 'improved_rmse_pixel', 'improved_mae_mm_weighted_sum']

    if config.decision_metric in config.criterion.minimize:
        config.criterion.state = 'min'
    elif config.decision_metric in config.criterion.maximize:
        config.criterion.state = 'max'
    else:
        raise

    config.device = torch.device('cuda')

    return config