import numpy as np
import torch
import torch_geometric.transforms as T
from pgn.data.dmpnn_utils import MolGraphTransform

import os
import os.path as osp
import shutil

# Content: Index in the bond_feature which holds distance information
DISTANCE_INDEX = 0

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

class LigandOnlyPretransform(object):
    """
    Transform object for ProximityGraphDataset to applied in the pre-transform step. Takes a proximity graph and removes
    any interaction edges/protein atoms from the graph.
    """
    def __call__(self, data):
        device = data.x.device
        x = data.x.to_numpy
        num_nodes = x.shape[0]
        ligand_index = np.arange(num_nodes)[data.x[:, -1] == 1]
        print(ligand_index)


def normalize_targets(dataset, index=None, mean=None, std=None, yield_stats=True):
    """
    Normalizes the training target to have mean 0 and stddev 1
    :param dataset: dataset to normalize the targets for
    :param index: Index into dataset for the subset of dataset to calculate statistics on for normalization
    :param mean: external mean to use for normalization (i.e. test set normalization)
    :param std: external stddev to use for normalization (i.e. test set normalization)
    :param yield_stats: toggle to yield [dataset, (mean, std)] if True or just [dataset] if False
    :return: A dataset with normalized targets
    """
    if index is None:
        index = np.arange(len(dataset.data.name))
    if mean is None:
        mean = dataset.data.y[list(index)].mean()
    if std is None:
        std = std = dataset.data.y[list(index)].std()


    dataset.data.y = (dataset.data.y - mean) / std

    if not yield_stats:
        return [dataset.data.y]
    else:
        return [dataset.data.y, (mean, std)]


def normalize_distance(dataset, args=None, index=None, mean=None, std=None, yield_stats=True):
    """
    Normalizes the training target to have mean 0 and stddev 1
    :param dataset: dataset to normalize the targets for
    :param index: Index into dataset for the subset of dataset to calculate statistics on for normalization
    :param mean: external mean to use for normalization (i.e. test set normalization)
    :param std: external stddev to use for normalization (i.e. test set normalization)
    :param yield_stats: toggle to yield [dataset, (mean, std)] if True or just [dataset] if False
    :return: A dataset with normalized targets
    """
    if index is None:
        index = np.arange(len(dataset.data.name))
    if mean is None:
        mean = dataset.data.edge_attr[list(index), 0].mean()

    if std is None:
        std = dataset.data.edge_attr[list(index), 0].std()

    if args is not None and args.encoder_type == 'dmpnn':
        for molgraph in dataset.data.molgraph:
            molgraph.apply_dist_norm(args.node_dim + DISTANCE_INDEX, float(mean), float(std))

    dataset.data.edge_attr[:, 0] = (dataset.data.edge_attr[:, 0] - mean) / std

    if not yield_stats:
        return [dataset.data.edge_attr]
    else:
        return [dataset.data.edge_attr, (mean, std)]



def parse_transforms(transforms):
    valid_transforms = {'one_hot': OneHotTransform(), 'molgraph': MolGraphTransform()}
    return T.Compose([valid_transforms.get(t) for t in transforms if t in valid_transforms.keys()])


def format_data_directory(args):
    data_path = args.data_path
    if len([d for d in os.listdir(data_path) if d[0] != '.']) != 0:
        raise ValueError("--data_path points to a non-empty directory. If the working_data has already been preprocessed. Please"
                         "use the --construct_graphs False argument")
    os.mkdir(osp.join(data_path, 'raw'))
    os.mkdir(osp.join(data_path, 'processed'))

    os.mkdir(osp.join(data_path, 'raw', 'train'))
    os.mkdir(osp.join(data_path, 'processed', 'train'))

    if args.split_type == 'defined':
        os.mkdir(osp.join(data_path, 'raw', 'test'))
        os.mkdir(osp.join(data_path, 'processed', 'test'))


def split_test_graphs(graph_path, test_list):
    """
    Splits out the core set paths and the train set from the graphs output from generate_all_graphs.
    :param graph_path: The path where the graph generation placed all graphs
    :param test_list: The path to the csv containing the graphs
    :return: None
    """
    with open(test_list, 'r') as f:
        test_ids = set(f.read().lower().split(','))
    directories = os.listdir(osp.join(graph_path, 'raw', 'train'))
    for dir in directories:
        if dir in test_ids:
            shutil.move(os.path.join(graph_path, 'raw', 'train', dir), os.path.join(graph_path, 'raw', 'test'))
