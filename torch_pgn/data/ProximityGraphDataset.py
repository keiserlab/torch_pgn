import os
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

from tqdm import tqdm

from torch_pgn.data.data_utils import LigandOnlyPretransform, RemoveProximityEdgesPretransform

"""
Code to load in the processed InteractionGraphs to pytorch geometric.
"""
# TODO: Convert to work with args object
class ProximityGraphDataset(InMemoryDataset):
    def __init__(self, args, transform=None, pre_transform=None):
        # TODO: Add docstring
        self.include_dist = args.include_dist
        self.enable_interacting_mask = args.enable_interacting_mask
        self.enable_molgraph = args.enable_molgraph
        self.args = args
        self.transform = transform
        self.pre_transform = pre_transform
        self.parse_pre_transforms()
        self.mode = 'train'
        super(ProximityGraphDataset, self).__init__(args.data_path, self.transform, self.pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if not self.include_dist:
            self.data.edge_attr = self.data.edge_attr[:, 1:]




    @property
    def raw_file_names(self):
        return [self.mode]

    @property
    def processed_file_names(self):
        return [self.mode + '/working_data.pt']

    def download(self):
        pass

    def parse_pre_transforms(self):
        if self.args.interaction_edges_removed:
            self.pre_transform = RemoveProximityEdgesPretransform()
        if self.args.ligand_only:
            self.pre_transform = LigandOnlyPretransform()

    def process(self):
        data_list = []
        base_dir = os.path.join(self.raw_dir, self.raw_paths[0])
        # TODO: enable multiprocessing support
        for subdir in tqdm(os.listdir(base_dir)):
            if subdir[0] != '.':
                x_path = os.path.join(base_dir, subdir, subdir + "_node_features.npy")
                edge_path = os.path.join(base_dir, subdir, subdir + "_edges.npy")
                label_path = os.path.join(base_dir, subdir, subdir + "_label.npy")
                edge_attr_path = os.path.join(base_dir, subdir, subdir + "_edge_features.npy")
                pos_path = os.path.join(base_dir, subdir, subdir + "_pos3d.npy")
                x = torch.from_numpy(np.load(x_path)).type(torch.FloatTensor)
                edge_index = np.load(edge_path)
                interacting_mask = None
                if self.enable_interacting_mask:
                    interacting_index = np.unique(edge_index)
                    interacting_mask = np.zeros(list(x.size())[0], dtype=bool)
                    interacting_mask[interacting_index] = True
                    interacting_mask = torch.from_numpy(interacting_mask).type(torch.BoolTensor)
                edge_index = torch.from_numpy(edge_index).type(torch.LongTensor)
                edge_attr = np.load(edge_attr_path)
                edge_attr = torch.from_numpy(edge_attr).type(torch.FloatTensor)
                pos = torch.from_numpy(np.load(pos_path)).type(torch.FloatTensor)
                y = torch.from_numpy(np.load(label_path).astype(np.float)).type(torch.FloatTensor)
                molgraph = (x.numpy().shape[0], edge_attr.numpy().shape[0], edge_index.numpy().shape[1])
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y, name=subdir,
                            mask=interacting_mask, molgraph=molgraph)
                data_list.append(data)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
