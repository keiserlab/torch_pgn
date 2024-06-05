import os
import os.path as osp
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

from tqdm import tqdm

"""
Code to load in the processed InteractionGraphs to pytorch geometric.
"""
# TODO: Convert to work with args object
class FingerprintDataset(InMemoryDataset):
    def __init__(self, args, transform=None, pre_transform=None):
        # TODO: Add docstring
        self.mode = 'train'
        super(FingerprintDataset, self).__init__(args.data_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.mode]

    @property
    def processed_file_names(self):
        return [self.mode + '/working_data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        base_dir = os.path.join(self.raw_dir, self.raw_paths[0])
        names = np.load(osp.join(base_dir, 'names.npy'))
        labels = np.load(osp.join(base_dir, 'labels.npy'))
        fps = np.load(osp.join(base_dir, 'fps.npy'))
        for idx in tqdm(range(len(names))):
            fp = torch.from_numpy(fps[idx, :]).type(torch.FloatTensor)
            y = torch.from_numpy(np.array([labels[idx]])).type(torch.FloatTensor)
            name = names[idx]
            data = Data(x=fp, y=y, name=name)
            data_list.append(data)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
