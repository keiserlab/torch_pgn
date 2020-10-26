import numpy as np
import torch
import torch_geometric.transforms as T
from pgn.data.dmpnn_utils import MolGraphTransform

import os
import os.path as osp

class OneHotTransform(object):
    """
    Transform object for ProximityGraphDataset with atomic number feature. Transforms the int feature into a 1-hot
    encoding that used atomic number (up to 100) as an index.
    #TODO: make it compatible with other atomic number index positions.
    """
    def __call__(self, data, encoder=None):
        device = data.x.device
        atom_type = data.x[:, 0].numpy().astype(int)
        one_hots = np.zeros((atom_type.shape[0], 100))
        for i, row in enumerate(one_hots):
            one_hots[i, atom_type[i]] += 1
        data.x = np.hstack((one_hots, data.x[:, 1:]))
        data.x = torch.from_numpy(data.x).type(torch.FloatTensor).to(device)
        return data


def normalize_targets(dataset, mean=None, std=None, yield_stats=False):
    """
    Normalizes the training target to have mean 0 and stddev 1
    :param dataset: dataset to normalize the targets for
    :param mean: external mean to use for normalization (i.e. test set normalization)
    :param std: external stddev to use for normalization (i.e. test set normalization)
    :param yield_stats: toggle to yield [dataset, (mean, std)] if True or just [dataset] if False
    :return: A dataset with normalized targets
    """
    if mean is None:
        mean = dataset.data.y.mean()
    if std is None:
        std = dataset.data.y.std()

    dataset.data.y = (dataset.data.y - mean) / std

    if yield_stats:
        return [dataset.data.y]
    else:
        return [dataset.data.y, (mean, std)]


def normalize_distance(dataset, mean=None, std=None, yield_stats=False):
    """
    Normalizes the training target to have mean 0 and stddev 1
    :param dataset: dataset to normalize the targets for
    :param mean: external mean to use for normalization (i.e. test set normalization)
    :param std: external stddev to use for normalization (i.e. test set normalization)
    :param yield_stats: toggle to yield [dataset, (mean, std)] if True or just [dataset] if False
    :return: A dataset with normalized targets
    """
    if mean is None:
        mean = dataset.data.edge_attr[:, 0].mean()
    if std is None:
        std = dataset.data.edge_attr[:, 0].std()

    dataset.data.edge_attr[:, 0] = (dataset.data.edge_attr[:, 0] - mean) / std

    if yield_stats:
        return [dataset.data.edge_attr[:, 0]]
    else:
        return [dataset.data.edge_attr[:, 0], (mean, std)]


def parse_transforms(transforms):
    valid_transforms = {'one_hot': OneHotTransform(), 'molgraph': MolGraphTransform()}
    return T.Compose([valid_transforms.get(t) for t in transforms if t in valid_transforms.keys()])


def format_data_directory(args):
    data_path = args.data_path
    os.mkdir(osp.join(data_path, 'raw'))
    os.mkdir(osp.join(data_path, 'processed'))

    os.mkdir(osp.join(data_path, 'raw', 'train'))
    os.mkdir(osp.join(data_path, 'processed', 'train'))

    if args.split_type == 'defined':
        os.mkdir(osp.join(data_path, 'raw', 'test'))
        os.mkdir(osp.join(data_path, 'processed', 'test'))
