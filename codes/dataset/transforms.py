import torch


from albumentations import (
    Cutout, HorizontalFlip,
    IAAAdditiveGaussianNoise, GaussNoise,
    IAASharpen,  RandomBrightnessContrast, OneOf, Compose, KeypointParams, BboxParams, Resize
)
import numpy as np

def strong_aug2(p, img_size):
    return Compose([
        Cutout(num_holes=50, max_h_size=2, max_w_size=2, fill_value=0, p=p[0]),
        HorizontalFlip(p=p[1]),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=p[2]),
        OneOf([
            IAASharpen(),
            RandomBrightnessContrast(),
        ], p=p[3]),
        Resize(img_size[0], img_size[1]),
    ], keypoint_params=KeypointParams(format='xy'),
       bbox_params=BboxParams(format='coco', label_fields=['category_ids']))

def inference_aug(img_size):
    return Compose([
        Resize(img_size[0], img_size[1]),
    ], keypoint_params=KeypointParams(format='xy'),
       bbox_params=BboxParams(format='coco', label_fields=['category_ids']))

def bbox_normalize(bbox:np.array, column:float, row:float):
    # x (column), y (row), width (column), height (row)
    bbox[:, 0] = bbox[:, 0] / column
    bbox[:, 1] = bbox[:, 1] / row
    bbox[:, 2] = bbox[:, 2] / column
    bbox[:, 3] = bbox[:, 3] / row

    return bbox

def points_normalize(points:np.array, dim0:float, dim1:float):
    points[..., 0] = points[..., 0] / dim0
    points[..., 1] = points[..., 1] / dim1

    return points


def get_weighted_sum2(pred):
    pred_col = torch.sum(pred, (-2))  # bach, c, column
    pred_row = torch.sum(pred, (-1))  # batch, c, row
    mesh_c = torch.arange(pred_col.shape[-1]).unsqueeze(0).unsqueeze(0).to(pred.device)
    mesh_r = torch.arange(pred_row.shape[-1]).unsqueeze(0).unsqueeze(0).to(pred.device)
    coord_c = torch.sum(pred_col * mesh_c, (-1)) / torch.sum(pred_col, (-1))
    coord_r = torch.sum(pred_row * mesh_r, (-1)) / torch.sum(pred_row, (-1))
    coord = torch.stack((coord_r, coord_c), -1)
    return coord

def make_gaussian(mean, size, std):
    mean = mean.unsqueeze(1).unsqueeze(1)
    var = std ** 2 # 64, 1
    grid = torch.stack(torch.meshgrid([torch.arange(size[0]), torch.arange(size[1])]), dim=-1).unsqueeze(0)
    grid = grid.to(mean.device)
    x_minus_mean = grid - mean  # 13, 1024, 1024, 2

    # (x-u)^2: (13, 512, 512, 2)  inverse_cov: (1, 1, 1, 1) > (13, 512, 512)
    gaus = (-0.5 * (x_minus_mean.pow(2) / var)).sum(-1).exp()

    # (13, 512, 512)
    return normalize_max2one(gaus)


def normalize_max2one(heatmap): #201017 #13 1024 1024  b 13 1024 1024
    max_val = heatmap.max(-2, True)[0].max(-1, True)[0]
    max_val[max_val==0] += 1e-10
    return heatmap / max_val
