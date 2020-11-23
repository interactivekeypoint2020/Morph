
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
from dataset.util import get_optflow
from dataset import transforms


class HintEncoder(nn.Module):
    def __init__(self,):
        super(HintEncoder, self).__init__()
        self.convs = nn.Sequential(
            self.make_block(77,64,3,2,1),
            self.make_block(64,64,3,1,1),
            self.make_block(64,64,3,2,1),
            self.make_block(64,128,3,1,1),
            self.make_block(128,128,3,1,1),
            self.make_block(128,128,3,1,1)
        )

    def make_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.convs(x)


class Ours(nn.Module):
    def __init__(self, config):
        super(Ours, self).__init__()
        self.config = config

        self.enc_conv = HintEncoder()

        self._init_unet()

        self.bbox_predictor = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, config.n_bbox, kernel_size=3, padding=1),
            nn.ReLU(),
            flatten_feature(),
            nn.Linear((config.image_size[0] // 16) * (config.image_size[1] // 16), 4)
        )

    def forward(self, input, hint_layer_flag=False):
        if input['hint_indices'] is None:
            hint_indices = None
        elif isinstance(input['hint_indices'], torch.Tensor):
            hint_indices = input['hint_indices'].data.cpu().tolist()
        else:
            hint_indices = list(input['hint_indices'])

        points = input['points_coord_gt'] # (batch, 13, 2)
        if hint_indices is not None and type(hint_indices[0]) != int:
            enc_heatmap = torch.zeros_like(input['points'])  # batch, 13, 512, 512
            for i in range(enc_heatmap.shape[0]):  # batch
                    for j in hint_indices[i]:
                        enc_heatmap[i, j] = transforms.make_gaussian(points[i, [j]], self.config.image_size,
                                                                       self.config.hint_std).float().squeeze(0)  # 512, 512

        else:
            enc_heatmap = torch.zeros_like(input['points'])

        img_with_anno =  input['img']

        #l1
        x_enc_1 = self.enc_1(img_with_anno)
        l1 = self.down_1(img_with_anno) # b, 64, 256, 256

        #l4
        x_enc_2 = self.enc_2(l1)
        l4 = self.down_2(x_enc_2) # b, 64, 128, 128
        x_enc_3 = self.enc_3(l4)
        l4 = self.down_3(x_enc_3) # b, 128, 64, 64

        #hint encoder

        l1_first = torch.cat(
            (l1, F.interpolate(enc_heatmap, size=l1.size()[2:], mode='bilinear', align_corners=True)),
            dim=1)  # torch.Size([b, 13+64, 256, 256])

        l1_first = self.enc_conv(l1_first)  #13+64,

        l4 = torch.cat((l1_first, l4), dim=1)  # 4, 128+128, 64, 64]

        #concat
        x = self.bottleneck(l4)  # 128

        if self.config.loss_weights[0] > 0.0:
            bbox = self.bbox_predictor(x)
        else:
            bbox = None


        x = self.up_1(x)
        x = self.dec_1(torch.cat((x_enc_3, x), 1))

        x = self.up_2(x)
        x = self.dec_2(torch.cat((x_enc_2, x), 1))

        x = self.up_3(x)  # b, 32, 512, 512 # b, 32, 1024, 1024
        x = self.dec_3(torch.cat((x_enc_1, x), 1))

        x = self.out_conv(x)

        x = x.sigmoid()

        if self.config.opt.flag and self.config.loss_weights[1] > 0.0:
            opts = self.opt_weighted_sum2(x)
        else:
            opts = None

        return bbox, x, opts

    def get_first(self, hint_index):
        p = self.config.hint.enc_dist[hint_index]
        p = (p / p.sum()).tolist()
        enc_idx = np.random.choice(hint_index, 1, p=p)
        return enc_idx

    def _init_unet(self):
        base_resnet = [*torchvision.models.resnet18(pretrained=True).children()]
        self.enc_1 = self._block(in_channels=3, out_channels=64, name='enc_1')
        self.down_1 = nn.Sequential(*base_resnet[:3])

        self.enc_2 = self._block(in_channels=64, out_channels=64, name='enc_2')
        self.down_2 = nn.Sequential(*base_resnet[3:5])

        self.enc_3 = self._block(in_channels=64, out_channels=64, name='enc_3')
        self.down_3 = nn.Sequential(*base_resnet[5])

        self.bottleneck = self._block(in_channels=128+128, out_channels=128, name='bottleneck')

        self.up_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_1 = self._block(in_channels=64 + 64, out_channels=64, name='dec_1')

        self.up_2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec_2 = self._block(in_channels=64 + 64, out_channels=64, name='dec_2')

        self.up_3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_3 = self._block(in_channels=32 + 64, out_channels=32, name='dec_3')

        self.out_conv = nn.Conv2d(in_channels=32, out_channels=self.config.n_class, kernel_size=1, padding=0)

    def opt_weighted_sum2(self, pred):
        # pred: batch, 13, 1024, 1024 (batch, c, row, column)

        pred_col = torch.sum(pred, (-2))  # bach, c, column
        pred_row = torch.sum(pred, (-1))  # batch, c, row

        # 1, 1, 1024
        mesh_c = torch.arange(pred_col.shape[-1]).unsqueeze(0).unsqueeze(0).to(pred.device)
        mesh_r = torch.arange(pred_row.shape[-1]).unsqueeze(0).unsqueeze(0).to(pred.device)

        # batch, 13
        coord_c = torch.sum(pred_col * mesh_c, (-1)) / torch.sum(pred_col, (-1))
        coord_r = torch.sum(pred_row * mesh_r, (-1)) / torch.sum(pred_row, (-1))

        # batch, 13, 2 (row, column)
        coord = torch.stack((coord_r, coord_c), -1)

        # normalize
        coord = transforms.points_normalize(coord, dim0=pred.size(-2), dim1=pred.size(-1))
        opts = get_optflow(coord, self.config, dtype='torch')

        return opts



    def _block(self, in_channels, out_channels, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv_1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm_1", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "relu_1", nn.ReLU(inplace=True)),
                    (
                        name + "conv_2",
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm_2", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "relu_2", nn.ReLU(inplace=True)),
                ]
            )
        )



class flatten_feature(nn.Module):
    def __init__(self, mode='def'):
        super(flatten_feature, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], x.shape[1], -1)