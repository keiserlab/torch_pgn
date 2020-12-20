import os
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from tqdm import tqdm

"""
Code to load in the processed InteractionGraphs to pytorch geometric.
"""

class ProximityGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, device='cpu', mode='train', include_dist=False,
                 enable_interacting_mask=False, enable_molgraph=False, load_index=None):
        # TODO: Add docstring
        self.mode = mode
        self.include_dist = include_dist
        self.enable_interacting_mask = enable_interacting_mask
        self.enable_molgraph = enable_molgraph
        super(ProximityGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data.to(device)
        self.load_index = load_index


    @property
    def raw_file_names(self):
        return [self.mode]

    @property
    def processed_file_names(self):
        return [self.mode + '/data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        base_dir = os.path.join(self.raw_dir, self.raw_paths[0])
        # TODO: enable multiprocessing support
        if self.load_index is None:
            subdir_list = os.listdir(base_dir)
        else:
            subdir_list = self.load_index
        for subdir in tqdm(subdir_list):
            if subdir[0] != '.':
                print(subdir)
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
                if not self.include_dist:
                    edge_attr = np.load(edge_attr_path)[:, 1:]
                else:
                    edge_attr = np.load(edge_attr_path)
                edge_attr = torch.from_numpy(edge_attr).type(torch.FloatTensor)
                pos = torch.from_numpy(np.load(pos_path)).type(torch.FloatTensor)
                y = torch.from_numpy(np.load(label_path).astype(np.float)).type(torch.FloatTensor)
                molgraph = None
                if self.enable_molgraph:
                    molgraph = (x.numpy(), edge_attr.numpy(), edge_index.numpy())
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y, name=subdir,
                            mask=interacting_mask, molgraph=molgraph)
                data_list.append(data)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
