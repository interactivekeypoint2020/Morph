import torch
from misc import metric
from tqdm import tqdm
from misc.util import argmax_heatmap
from dataset import transforms


def run(config, model, grad:bool, criterion, loader, optimizer=None, writer=None):
    if grad:
        iter_flag = [0]

        model.train()
        model, optimizer, results = iter_epoch(config=config, model=model, grad=grad, criterion=criterion, loader=loader, optimizer=optimizer, writer=writer, iter_flag=iter_flag)
        return model, optimizer, results
    else:
        iter_flag = [0]

        model.eval()
        with torch.no_grad():
            results = iter_epoch(config=config, model=model, grad=grad, criterion=criterion, loader=loader, writer=writer, iter_flag=iter_flag)

        return results


def iter_epoch(config, model, grad, criterion, loader, optimizer=None, writer=None, iter_flag=None):
    outs = {'bbox':[], 'keypoint':[], 'weighted_sum_keypoint':[]}
    labels = {'bbox':[], 'keypoint':[]}
    pspace_batch = []

    loader = tqdm(loader, bar_format='{desc:<40}{percentage:3.0f}%|{bar:20}{r_bar}')
    for batch_idx, batch in enumerate(loader):
        pspace_batch.append(batch['pspace'])
        batch['img'] = batch['img'].to(config.device)

        print_loss = {}
        for mode_idx in iter_flag:
            hint_layer_flag = None
            _out = model(batch, hint_layer_flag)

            out = {'bbox': _out[0], 'points': _out[1], 'opts': _out[2]}

            loss, _print_loss = criterion(out, batch, mode_idx=mode_idx)

            for key in _print_loss:
                print_loss[key] = _print_loss[key] #
            if grad:
                optimizer.update_model(loss, mode_idx=mode_idx)

        if batch_idx == 0:
            losses = {key:[] for key in print_loss}
        for loss_key in print_loss:
            losses[loss_key].append(print_loss[loss_key])


        loader.set_description("loss: {:.5f}".format(loss.item()))
        if grad and writer:
            writer.write_iter(**print_loss)
        if out['bbox'] != None:
            outs['bbox'].append(out['bbox'].data.cpu())
            labels['bbox'].append(batch['bbox'].data.cpu())
        out_keypoint = argmax_heatmap(out['points']).data.cpu()
        outs['keypoint'].append(out_keypoint)
        labels['keypoint'].append(batch['points_coord_gt'].data.cpu())
        if 'weighted_sum' in config.metrics:
            outs['weighted_sum_keypoint'].append(transforms.get_weighted_sum2(out['points']).data.cpu())



    if out['bbox'] != None:
        outs['bbox'] = torch.cat(outs['bbox']).view(-1,4)
        labels['bbox'] = torch.cat(labels['bbox']).view(-1,4)

    outs['keypoint'] = torch.cat(outs['keypoint'])
    labels['keypoint'] = torch.cat(labels['keypoint'])
    if 'weighted_sum' in config.metrics:
        outs['weighted_sum_keypoint'] = torch.cat(outs['weighted_sum_keypoint'])

    pspace_batch = torch.cat(pspace_batch)
    results = metric.calculate_metric(config, outs, labels, losses, pspace_batch)


    if grad:
        return model, optimizer, results
    else:
        return results