import json
import torch

def get_optflow(points, config, dtype='nobatch'):

    dist_pairs = torch.tensor(config.opt.pairs[0])
    unit_vec_pairs = torch.tensor(config.opt.pairs[1])

    # dist
    if dtype == 'nobatch':  # 13, 2 (row, col)
        from_vec = points[dist_pairs[:, 0]]
        to_vec = points[dist_pairs[:, 1]]
    else:  # batch, 13, 2
        from_vec = points[:, dist_pairs[:, 0]]
        to_vec = points[:, dist_pairs[:, 1]]

    diffs = to_vec - from_vec
    dists = torch.norm(diffs, dim = -1).unsqueeze(-1) # (batch, 16,1) or (16,1)
    dists = torch.cat((dists, torch.zeros_like(dists)), dim=-1) # (batch, 16, 2)

    # unit_vec_pairs
    if dtype == 'nobatch':  # 13, 2 (row, col)
        from_vec = points[unit_vec_pairs[:, 0]]
        to_vec = points[unit_vec_pairs[:, 1]]
    else:  # batch, 13, 2
        from_vec = points[:, unit_vec_pairs[:, 0]]
        to_vec = points[:, unit_vec_pairs[:, 1]]
    diffs = to_vec - from_vec

    unit_vecs = diffs / torch.norm(diffs, dim = -1).unsqueeze(-1) * config.opt_unit_dist_unit_vector_size

    opts = torch.cat((dists, unit_vecs), dim=-2) # (batch, (dist num + unitvec num), 2)



    return opts

def get_split_data(config):
    with open(config.PATH.DATA.TRAIN, 'r') as f:
        train_data = json.load(f)

    with open(config.PATH.DATA.VAL, 'r') as f:
        val_data = json.load(f)

    with open(config.PATH.DATA.TEST, 'r') as f:
        test_data = json.load(f)

    return train_data, val_data, test_data
