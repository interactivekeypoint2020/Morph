import dataset.transforms as transforms
from dataset.util import *
import numpy as np
import copy
import os

def load_png(img_path, size=None):
    from PIL import Image
    img = Image.open(img_path)
    column, row = img.size
    img = np.array(img)
    return img, row, column

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split:str, data:list):

        # init
        self.config = config
        self.split = split
        self.data = data

        if self.split == 'train' and self.config.aug.p:
           self.transformer = transforms.strong_aug2(self.config.aug.p, self.config.image_size)
        else:
           self.transformer = transforms.inference_aug(self.config.image_size)

        self.loadimage = load_png

        if self.config.debug:
            self.data = self.data[0:20]

        for item in self.data:
                item[self.config.DICT_KEY.BBOX] = [item[self.config.DICT_KEY.BBOX]]
                item[self.config.DICT_KEY.RAW_SIZE] = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # indexing
        item = self.data[index]

        # image load
        img_path = os.path.join(self.config.PATH.DATA.VERSION, item[self.config.DICT_KEY.IMAGE])
        img, row, column = self.loadimage(img_path, item[self.config.DICT_KEY.RAW_SIZE])

        # pixel spacing
        pspace_list = item[self.config.DICT_KEY.PSPACE] # row, column
        raw_size_and_pspace = torch.tensor([row, column] + pspace_list)

        # points load (13,2) (column, row)
        points = item[self.config.DICT_KEY.POINTS]

        # bbox load  [minx, miny, w, h] (x: column, y: row)
        bbox = item[self.config.DICT_KEY.BBOX]

        # data augmentation
        # from | img: (1670, 2010) PIL -> numpy,  points: (13) list, bbox: (4), list
        # to | img: (1024, 1024, 3) numpy,  points: (13, 2) list (column, row), bbox: (1, 4) list (column, row)
        transformed = self.transformer(image=img, keypoints=points, bboxes=bbox, category_ids=[1])
        img, points, bbox = transformed["image"], transformed["keypoints"], transformed["bboxes"]

        points = np.array(points)

        # np array to tensor
        img = torch.tensor(img, dtype=torch.float)
        img = img.permute(2, 0, 1)
        img /= 255.0

        # [minx, miny, w, h] -> [x1, y1, x2, y2] (x: column, y: row)
        bbox = torch.tensor(bbox, dtype=torch.float)
        bbox[:, 2] += bbox[:, 0]
        bbox[:, 3] += bbox[:, 1]
        bbox = transforms.bbox_normalize(bbox, row=self.config.image_size[0], column=self.config.image_size[1])

        # 13, 2 (1024 x 1024 unit) (column, row) > (row, column)
        points = torch.tensor(copy.deepcopy(points[:, ::-1]), dtype=torch.float)

        # offset vector between points (row, column)
        if self.config.opt.flag:
            points_norm = transforms.points_normalize(points.clone().detach(),
                                                      dim0=self.config.image_size[0], dim1=self.config.image_size[1])
            opts_gt = get_optflow(points_norm, self.config)
        else:
            opts_gt = torch.tensor([], dtype=torch.float)

        # make heatmap from points
        points_heatmap = transforms.make_gaussian(points, self.config.image_size, self.config.heatmap_std)
        points_heatmap = points_heatmap.float()

        # hint
        if self.config.use_hint:
            if self.split == 'train':
                hint = torch.zeros_like(points_heatmap)
                num_hint = np.random.choice(range(self.config.numkey ), size=None, p=self.config.hint.num_dist)
                hint_indices = np.random.choice(range(self.config.numkey ), size=num_hint, replace=False) #[1,2,3]




                hint_mask = np.zeros(self.config.numkey )
                hint_mask[hint_indices] = 1
                hint_mask = torch.tensor(hint_mask, dtype=torch.long)

            else:
                hint = torch.zeros_like(points_heatmap)
                hint_indices = -1
                hint_mask = torch.tensor([])
        else:
            hint = None
            hint_indices = None
            hint_mask = None

        return img_path, img, bbox, points_heatmap, hint, opts_gt, raw_size_and_pspace, hint_indices, points, hint_mask

def collate_fn(batch):
    batch = list(zip(*batch))

    if batch[4][0] is None:
        batch_4 = None
        batch_7 = None
        batch_9 = None
    else:
        batch_4 = torch.stack(batch[4])
        batch_7 = batch[7]
        batch_9 = torch.stack(batch[9])

    batch_dict = {
        'img_path':batch[0], # list
        'img':torch.stack(batch[1]),
        'bbox':torch.stack(batch[2]),
        'points':torch.stack(batch[3]),
        'hints':batch_4,
        'opts_gt': torch.stack(batch[5]),
        'pspace': torch.stack(batch[6]),
        'hint_indices': batch_7, # 길이가 다 달라서, list로 그냥 넣어줌
        'points_coord_gt': torch.stack(batch[8]),
        'hint_mask': batch_9
    }
    return batch_dict

def dataloader(config, split:str, data:list):

    # split : 'train' / 'val' / 'test'
    dataset = Dataset(config=config, split=split, data=data)
    drop_last = False
    # loader
    if split == 'train':
        if config.sampler is None:
            sampler = None
            shuffle = True
        else:
            raise NotImplementedError
        drop_last = True
    else:
        sampler = None
        shuffle = False

    def _init_fn(worker_id):
        np.random.seed(config.seed + worker_id)


    data_loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, sampler=sampler, worker_init_fn=_init_fn,
                              batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=collate_fn, drop_last=drop_last)
    return data_loader


def get_split_data(config, split=None):
    if split == 'train':
        with open(config.PATH.DATA.TRAIN, 'r') as f:
            train_data = json.load(f)
        return train_data
    elif split =='val':
        with open(config.PATH.DATA.VAL, 'r') as f:
            val_data = json.load(f)
        return val_data

    elif split =='test':
        with open(config.PATH.DATA.TEST, 'r') as f:
            test_data = json.load(f)
        return test_data

def get_dataloader(config, split):
    data = get_split_data(config, split)
    loader = dataloader(config, split, data)
    return loader
