from torch_pgn.models.pfp_encoder import PFPEncoder
from torch_pgn.models.dmpnn_encoder import MPNEncoder
from torch_pgn.models.GGNet import GGNet
from torch_pgn.models.FPEncoder import FPEncoder
from torch_pgn.models.DimeNet import DimeNetPlusPlus
from torch_pgn.args import TrainArgs

import torch.nn as nn
from torch.nn import ReLU, Sequential, Linear, Dropout

class PGNNetwork(nn.Module):
    """A netork that includes the message passing PFPEncoder and a feed-forward network for learning tasks."""
    def __init__(self, args: TrainArgs, node_dim: int, bond_dim: int):
        """
        Initialization of the PFPNetwork model
        :param args: Combined arguments for the encoder and feed-forward network
        :param node_dim: Number of node features
        :param bond_dim: number of bond features
        """
        super(PGNNetwork, self).__init__()

        self.args = args
        self.node_dim = node_dim
        self.bond_dim = bond_dim

        self.construct_encoder()
        self.construct_feed_forward()

    def construct_encoder(self):
        """
        Constructs the message passing network for encoding proximity graphs.
        """
        if self.args.encoder_type == 'pfp':
            self.encoder = PFPEncoder(self.args, self.node_dim, self.bond_dim)
        elif self.args.encoder_type == 'dmpnn':
            self.encoder = MPNEncoder(self.args, self.node_dim, self.bond_dim)
        elif self.args.encoder_type == 'ggnet':
            self.encoder = GGNet(self.args, self.node_dim, self.bond_dim)
        elif self.args.encoder_type == 'dimenet++':
            self.encoder = DimeNetPlusPlus(self.args, self.node_dim)
        elif self.args.encoder_type == 'fp':
            self.encoder = FPEncoder(self.args)


    def construct_feed_forward(self):
        """
        Constructs the feed-forward network used for regression tasks
        """
        dropout_prob = self.args.dropout_prob
        input_dim = self.args.fp_dim
        hidden_dim = self.args.hidden_dim
        num_layers = self.args.num_layers
        num_classes = self.args.num_classes
        first_hidden_dim = self.args.ff_dim_1

        dropout = Dropout(dropout_prob)
        activation_fn = ReLU()

        if num_layers == 1:
            network = [dropout, Linear(first_hidden_dim, num_classes)]
        else:
            network = [dropout, Linear(input_dim, first_hidden_dim)]
            for i in range(num_layers - 2):
                network.extend([activation_fn, dropout, Linear(first_hidden_dim, hidden_dim)])
            network.extend([activation_fn, dropout, Linear(hidden_dim, num_classes)])
        self.feed_forward = Sequential(*network)

    def forward(self, data):
        """
        Runs the PFPNetwork on the input
        :param input: batch of Proximity Graphs
        :return: Output of the PFPNetwork
        """
        if self.args.encoder_type == 'dimenet++':
            out = self.encoder(node_feats=data.x, edge_index=data.edge_index, pos=data.pos, batch=data.batch)
        else:
            out = self.feed_forward(self.encoder(data))

        return out.view(-1)