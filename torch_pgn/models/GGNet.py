import torch
import torch.nn.functional as F
from torch.nn import ReLU, Sequential, Linear, BatchNorm1d, Dropout, GRU

from torch_geometric.nn import NNConv, Set2Set
from torch_pgn.args import TrainArgs


class GGNet(torch.nn.Module):
    """Stock network from QM9 prediction paper gilmer et. al"""
    def __init__(self, args: TrainArgs, node_dim: int, bond_dim: int):
        super(GGNet, self).__init__()

        self.node_dim = node_dim
        self.bond_dim = bond_dim
        self.device = args.device
        self.nn_conv_in_dim = args.nn_conv_in_dim
        self.nn_conv_internal_dim = args.nn_conv_internal_dim
        self.nn_conv_out_dim = args.nn_conv_out_dim
        self.nn_conv_aggr = args.nn_conv_aggr
        self.depth = args.depth
        self.ligand_only_readout = args.ligand_only_readout

        self.atom_expand_nn = Sequential(Linear(self.node_dim, self.nn_conv_in_dim), ReLU())

        nn = Sequential(Linear(self.bond_dim, self.nn_conv_internal_dim),
                        ReLU(),
                        Linear(self.nn_conv_internal_dim, self.nn_conv_out_dim))
        self.conv = NNConv(self.nn_conv_in_dim, self.nn_conv_in_dim, nn, aggr=self.nn_conv_aggr)
        self.gru = GRU(self.nn_conv_in_dim, self.nn_conv_in_dim)

        self.set2set = Set2Set(self.nn_conv_in_dim, processing_steps=3)

    def forward(self, data):
        out = self.atom_expand_nn(data.x)
        h = out.unsqueeze(0)

        for i in range(self.depth):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        if self.ligand_only_readout:
            ligand_message_mask = data.x[:, -1] == 1.
            print(out.size())
            batch = data.batch[ligand_message_mask]
            out = out[ligand_message_mask]
        else:
            batch = data.batch
        out = self.set2set(out, batch)

        return out