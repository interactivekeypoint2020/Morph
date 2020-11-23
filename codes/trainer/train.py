import os
import copy
import torch

from trainer.run import run
from misc import metric, util


def train(config, model, train_loader, val_loader, criterion, optimizer,writer=None, epoch_range=range(1, 1001), patience=0):
    best_params = None
    best_epoch = 0
    best_metric = metric.init_best_metric(config)
    patience = 0
    util.log(config, '================== Training started ===================')
    for epoch in epoch_range:
        # train
        model, optimizer, train_results = run(config=config, model=model, grad=True, criterion=criterion, loader=train_loader, optimizer=optimizer, writer=writer)

        # validation

        val_results = run(config=config, model=model, grad=False, criterion=criterion, loader=val_loader)

        # tensorboard
        if writer:
            writer.write_epoch(train_results, val_results, epoch)
            # TODO: tensorboard 이미지 출력

        sentence = 'Epoch {} - {}'.format(epoch, val_results)
        util.log(config, sentence=sentence)

        if epoch % 5 == 0:
            print('::: Version {} patience {} :::'.format(config.version, patience))

        # save
        dec_result = val_results[config.decision_metric]
        if config.scheduler is not None:
            optimizer.scheduler.step(dec_result)


        for save_standard in [20,10,5,1]:
            if not metric.compare(config=config, best_metric=save_standard, current_metric=dec_result):
                save_dict = {'model': copy.deepcopy(model.state_dict()),
                             'optimizer': optimizer,
                             'epoch': epoch,
                             'train_results': train_results,
                             'val_results': val_results,
                             'patience': patience}
                save_path = os.path.join(config.PATH.RUN.SAVE, 'model_{}.pth'.format(save_standard))
                if not config.debug:
                    torch.save(save_dict, save_path)

                util.log(config,
                         '===== Intermediate model save - Epoch [{}] Decision metric {} [{}] Save name [{}] ====='.format(epoch,
                                                                                                  config.decision_metric,
                                                                                                  dec_result,'model_{}.pth'.format(save_standard)))
                break



        if metric.compare(config=config, best_metric=best_metric, current_metric=dec_result):
            best_metric = dec_result
            best_params = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0

            save_dict = {'model': best_params,
                         'optimizer': optimizer,
                         'epoch': epoch,
                         'train_results': train_results,
                         'val_results': val_results,
                         'patience': patience}

            save_path = os.path.join(config.PATH.RUN.SAVE, 'model.pth')
            if not config.debug:
                torch.save(save_dict, save_path)

            util.log(config,
                     '=====  Model saved ::: Epoch [{}] Decision metric {} [{}] ====='.format(best_epoch, config.decision_metric,
                                                                                          best_metric))
        else:
            patience += 1

        if patience == config.patience:
            util.log(config, '===== Stopped early ::: Last saved Epoch [{}] Decision metric {} [{}] ====='.format(best_epoch, config.decision_metric,
                                                                                          best_metric))
            break


    util.log(config, '================== Training finished ===================')
    return best_epoch, best_params, best_metric