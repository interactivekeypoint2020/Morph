import torch
from torch import nn

def get_criterion(config):
    criterion = base_criterion(config)
    return criterion

class base_criterion(nn.Module):
    def __init__(self, config):
        super(base_criterion, self).__init__()
        self.config = config
        if config.loss == 'bce':
            self.criterion = nn.BCELoss()

        elif config.loss == 'cross':
            self.criterion = nn.CrossEntropyLoss()

        elif config.loss == 'CustomLoss2':
            self.criterion = CustomLoss2(config)

        else:
            raise

    def forward(self, out, batch, mode_idx=None):
        if self.config.loss == 'CustomLoss2':
            loss = self.criterion(out, batch, mode_idx=None)
        else:
            loss = self.criterion(out, batch)
        return loss

def get_opt_loss(config, loss, batch, out):
    criterion = nn.L1Loss()

    opt_loss = criterion(out['opts'], batch['opts_gt'].to(config.device))

    loss += opt_loss * config.loss_weights[1]
    return loss, opt_loss


class CustomLoss2(nn.Module):
    def __init__(self, config):
        super(CustomLoss2, self).__init__()
        self.config = config

    def forward(self, out, batch, mode_idx):
        points_label = batch['points'].to(self.config.device)

        points_loss = nn.BCELoss()(out['points'], points_label)
        loss = points_loss

        if self.config.loss_weights[0] > 0.0:
            bbox_label = batch['bbox'].to(self.config.device)
            bbox_loss = nn.L1Loss()(out['bbox'], bbox_label)
            loss += bbox_loss * self.config.loss_weights[0]
        else:
            bbox_loss = torch.tensor(-1)

        if self.config.opt.flag and self.config.loss_weights[1] > 0.0:
            loss, opt_loss = get_opt_loss(self.config, loss, batch, out)
        else:
            opt_loss = torch.tensor(-1)

        print_loss = {
            'loss': loss.item(),
            'bbox_loss':bbox_loss.item(),
            'heatmap_loss':points_loss.item(),
            'opt_loss':opt_loss.item()
        }
        return loss, print_loss