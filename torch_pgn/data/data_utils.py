import numpy as np
import torch
import torch_geometric.transforms as T
from torch_pgn.data.dmpnn_utils import MolGraphTransform

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
        x = data.x.numpy()
        edge_index = data.edge_index.numpy()
        edge_attr = data.edge_attr.numpy()
        num_nodes = x.shape[0]
        ligand_index = np.arange(num_nodes)[data.x[:, -1] == 1]
        x = x[ligand_index, :]
        edge_mask = np.isin(edge_index, ligand_index)
        edge_mask = np.logical_and(edge_mask[0, :], edge_mask[1, :])
        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask, :]
        data.molgraph = (x.shape[0], edge_attr.shape[0], edge_index.shape[1])
        x = torch.from_numpy(x).type(torch.FloatTensor)
        edge_index = torch.from_numpy(edge_index).type(torch.LongTensor)
        edge_attr = torch.from_numpy(edge_attr).type(torch.FloatTensor)
        data.x = x
        data.edge_index = edge_index
        data.edge_attr = edge_attr

        return data


class RemoveProximityEdgesPretransform(object):
    """
    Transform object for ProximityGraphDataset to applied in the pre-transform step. Takes a proximity graph and removes
    any interaction edges/protein atoms from the graph.
    """
    def __call__(self, data):
        x = data.x.numpy()
        edge_index = data.edge_index.numpy()
        edge_attr = data.edge_attr.numpy()
        num_nodes = x.shape[0]
        ligand_index = np.arange(num_nodes)[data.x[:, -1] == 1]
        protein_index = np.arange(num_nodes)[data.x[:, -1] == 0]
        ligand_edge_mask = np.isin(edge_index, ligand_index)
        ligand_edge_mask = np.logical_and(ligand_edge_mask[0, :], ligand_edge_mask[1, :])
        protein_edge_mask = np.isin(edge_index, protein_index)
        protein_edge_mask = np.logical_and(protein_edge_mask[0, :], protein_edge_mask[1, :])
        edge_mask = np.logical_or(ligand_edge_mask, protein_edge_mask)
        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask, :]
        data.molgraph = (x.shape[0], edge_attr.shape[0], edge_index.shape[1])
        edge_index = torch.from_numpy(edge_index).type(torch.LongTensor)
        edge_attr = torch.from_numpy(edge_attr).type(torch.FloatTensor)
        data.edge_index = edge_index
        data.edge_attr = edge_attr

        return data


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
        index = np.arange(len(dataset._data.name))
    if mean is None:
        mean = dataset._data.y[list(index)].mean()
    if std is None:
        std = dataset._data.y[list(index)].std()


    dataset._data.y = (dataset._data.y - mean) / std

    if not yield_stats:
        return [dataset._data.y]
    else:
        return [dataset._data.y, (mean, std)]


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
        index = np.arange(len(dataset._data.name))
    if mean is None:
        mean = dataset._data.edge_attr[list(index), 0].mean()

    if std is None:
        std = dataset._data.edge_attr[list(index), 0].std()

    if args is not None and args.encoder_type == 'dmpnn':
        for molgraph in dataset._data.molgraph:
            molgraph.apply_dist_norm(args.node_dim + DISTANCE_INDEX, float(mean), float(std))

    dataset._data.edge_attr[:, 0] = (dataset._data.edge_attr[:, 0] - mean) / std

    if not yield_stats:
        return [dataset._data.edge_attr]
    else:
        return [dataset._data.edge_attr, (mean, std)]



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


class RandomLigandTranslationTransform(object):
    """
    TODO: Not applied. Will go back latter to implement. Need to figure out DMPNN complications.
    Transform object for ProximityGraphDataset to applied in the pre-transform step. Takes a proximity graph and updates
    any interaction edges/protein atoms from the graph.
    """
    def __init__(self, mean=None, std=None, scale=0.05):
        """
        Instantiation with modifying parameters.
        :param mean: The mean to normalize the distances calculated with the translated
        :param std:
        :param scale:
        """
        self.mean = mean
        self.std = std
        self.scale = scale

    def __call__(self, data):
        """
        Take a data object and applied a random rigid-body translation of the average size scale to the ligand and recalculates
        proximity edges based off of this new position.
        :param data: Data to be transformed.
        :return: Transformed data object.
        """
        tranlation = np.random.normal(0., self.scale, (1,3))
        ligand_mask = data.x[:, -1] == 1
        data.pos[ligand_mask, :] += tranlation
        prox_edge_mask = data.edge_attr[:, -1] == 1
        prox_edges = data.edge_index[:, prox_edge_mask]
        pos1 = data.pos[prox_edges[0, :]]
        pos2 = data.pos[prox_edges[1, :]]
        dist = torch.sqrt(torch.sum(torch.pow((pos1 - pos2), 2)))
        if self.mean is not None:
            dist = (dist - self.mean) / self.std
        data.edge_attr[prox_edge_mask, 0] = dist
        return data
