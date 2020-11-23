import time
import sys
import os

import torch
import torch.nn as nn

from misc import util
from misc.config import get_config
from dataset.dataset import get_dataloader
from misc.criterion import get_criterion
from misc.optimizer import get_optimizer
from trainer.train import train
from trainer.test import test
from model.model import get_model
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main(config):
    if config.tensorboard:
        writer = util.writer(config)
    else:
        writer = None

    # model
    device_ids = list(range(len(config.gpu.split(','))))
    model = nn.DataParallel(get_model(config), device_ids=device_ids)
    model.to(config.device)

    n_params = 0
    for k,v in model.named_parameters():
        n_params += v.reshape(-1).shape[0]
    util.log(config, 'Number of model parameters : {}'.format(n_params))

    # optimizer, criterion
    optimizer = get_optimizer(config, model)
    criterion = get_criterion(config)
    if config.mode.train:
        # dataloader
        train_loader = get_dataloader(config, 'train')
        val_loader = get_dataloader(config, 'val')

        print('TRAIN START...')
        epoch_range = range(1, config.epochs+1)
        patience = 0
        best_epoch, best_params, best_metric = train(config=config, model=model, train_loader=train_loader, val_loader=val_loader,
                                                     criterion=criterion, optimizer=optimizer,
                                                     writer=writer, epoch_range=epoch_range, patience=patience)
        del train_loader
        del val_loader
    else:
        best_save_path = os.path.join(config.LOADPATH, 'model.pth')
        best_save = torch.load(best_save_path)
        best_params = best_save['model']
        best_epoch = best_save['epoch']
    util.log(config, '================= Version {}      ================'.format(config.version))



    if config.mode.val:
        config.batch_size = 1
        val_loader = get_dataloader(config, 'val')

        print('VALIDATION START...')
        model.load_state_dict(best_params)
        model.to(config.device)
        model.eval()
        if not config.mode.train:
            util.log(config, 'Best model at epoch {} loaded Decision metric {} {} loaded'.format(best_epoch, config.decision_metric,
                                                                                          best_save['val_results'][config.decision_metric]))
        test(config=config, model=model, test_loader=val_loader, criterion=criterion, writer=writer, split='val')
    util.log(config, '================= Version {}      ================'.format(config.version))

    if config.mode.test:
        config.batch_size = 1
        test_loader = get_dataloader(config, 'test')

        print('TEST START...')

        model.load_state_dict(best_params)
        model.to(config.device)
        model.eval()
        util.log(config, 'Best model at epoch {} loaded'.format(best_epoch))
        test(config=config, model=model, test_loader=test_loader, criterion=criterion, writer=writer, split='test')

    return

import datetime
if __name__ == '__main__':
    start_time = time.time()

    config = get_config()
    util.log(config, '================= Start | {} ================'.format(datetime.datetime.now()))
    util.log(config, '{}'.format(sys.argv))
    main(config)
    end_time = time.time()
    util.log(config,'================= End | {} {:.2f} hours ================'.format(datetime.datetime.now(),
                                                                                    (end_time-start_time)/3600))
    util.log(config, '================= Version {}      ================'.format(config.version))