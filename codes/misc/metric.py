import numpy as np
import torch
import torch.nn as nn
import copy

def pixel2mm(config, points, pspace):
    points[:, :, 1] = points[:, :, 1] / config.image_size[1] * pspace[:, 1:2] * pspace[:, 3:4]
    points[:, :, 0] = points[:, :, 0] / config.image_size[0] * pspace[:, 0:1] * pspace[:, 2:3]

    return points

def calculate_metric(config, outs, labels, losses, pspace):
    results = {}

    for loss in losses:
        results[loss] = np.mean(losses[loss])

    out_keypoint = pixel2mm(config, copy.deepcopy(outs['keypoint'].detach()), pspace)
    label_keypoint = pixel2mm(config, copy.deepcopy(labels['keypoint'].detach()), pspace)

    if 'mae' in config.metrics:
        results['mae_pixel'] = nn.L1Loss()(outs['keypoint'], labels['keypoint']).item()
        results['mae_mm'] = nn.L1Loss()(out_keypoint, label_keypoint).item()




    if 'rmse' in config.metrics:
        results['rmse_pixel'] = torch.sqrt(nn.MSELoss()(outs['keypoint'], labels['keypoint'])).item()
        results['rmse_mm'] = torch.sqrt(nn.MSELoss()(out_keypoint, label_keypoint)).item()



    if 'iou' in config.metrics:
        results['iou'] = iou_metric(outs['bbox'], labels['bbox'])

    if 'weighted_sum' in config.metrics:
        weighted_sum_keypoint_mm = pixel2mm(config, copy.deepcopy(outs['weighted_sum_keypoint'].detach()), pspace)
        results['mae_pixel_weighted_sum'] = nn.L1Loss()(outs['weighted_sum_keypoint'], labels['keypoint']).item()
        results['mae_mm_weighted_sum'] = nn.L1Loss()(weighted_sum_keypoint_mm, label_keypoint).item()

        if 'mre' in config.metrics:
            y_diff_sq = (weighted_sum_keypoint_mm[:, :, 0] - label_keypoint[:, :, 0]) ** 2
            x_diff_sq = (weighted_sum_keypoint_mm[:, :, 1] - label_keypoint[:, :, 1]) ** 2
            sqrt_x2y2 = torch.sqrt(y_diff_sq + x_diff_sq)  # (batch,13)
            results['mre_mm_weighted_sum'] = torch.mean(sqrt_x2y2).item()

        if 'improved_mae_mm_weighted_sum' in config.metrics:
            results['improved_mae_mm_weighted_sum'] = config.loaded_hint0_metric - results['mae_mm_weighted_sum']

    if 'mre' in config.metrics:
        # (batch, 13, 2)
        y_diff_sq = (out_keypoint[:,:,0] - label_keypoint[:,:,0]) **2
        x_diff_sq = (out_keypoint[:,:,1] - label_keypoint[:,:,1]) **2
        sqrt_x2y2 = torch.sqrt(y_diff_sq+x_diff_sq) # (batch,13)
        results['mre_mm'] = torch.mean(sqrt_x2y2).item()

    return results


def init_best_metric(config):
    if config.decision_metric in config.criterion.minimize:
        # smaller better
        best_metric = 1e4
    elif config.decision_metric in config.criterion.maximize:
        # larger better
        best_metric = -1000
    else:
        print('ERROR: NOT SPECIFIED DECISION METRIC {}'.format(config.decision_metric))
        raise

    return best_metric

def compare(config, best_metric, current_metric):

    if config.decision_metric in config.criterion.minimize:
        if best_metric > current_metric:
            return True
    elif config.decision_metric in config.criterion.maximize:
        if best_metric < current_metric:
            return True
    else:
        print('ERROR: NOT SPECIFIED DECISION METRIC {}'.format(config.decision_metric))
        raise

    return False


def iou_metric(a, b, mean=True):
    # a, b : torch tensor / size : (batch, 4) --- upper left (x, y, w, h)

    stack = torch.stack((a, b), dim=2)
    x_left = torch.max(stack[:, 0], dim=1)[0]
    x_right = torch.min(stack[:, 0] + stack[:, 2], dim=1)[0]
    y_upper = torch.min(stack[:, 1], dim=1)[0]
    y_lower = torch.max(stack[:, 1] - stack[:, 3], dim=1)[0]
    inter_Area = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_upper - y_lower, min=0)

    box_a_Area = a[:, 2] * a[:, 3]
    box_b_Area = b[:, 2] * b[:, 3]
    iou = inter_Area / (box_a_Area + box_b_Area - inter_Area)

    return torch.sum(iou).item() / len(iou)