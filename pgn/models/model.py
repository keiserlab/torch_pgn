from pgn.models.pfp_encoder import PFPEncoder
from pgn.args import TrainArgs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU, Sequential, Linear, Dropout

class PFPNetwork(nn.Module):
    """A netork that includes the message passing PFPEncoder and a feed-forward network for learning tasks."""
    def __init__(self, args: TrainArgs, node_dim: int, bond_dim: int):
        """
        Initialization of the PFPNetwork model
        :param args: Combined arguments for the encoder and feed-forward network
        :param node_dim: Number of node features
        :param bond_dim: number of bond features
        """
        super(PFPNetwork, self).__init__()

        self.args = args
        self.node_dim = node_dim
        self.bond_dim = bond_dim

        self.construct_encoder()

    def construct_encoder(self):
        """
        Constructs the message passing network for encoding proximity graphs.
        """
        self.encoder = PFPEncoder(self.args, self.node_dim, self.bond_dim)


    def construct_feed_forward(self):
        """
        Constructs the feed-forward network used for regression tasks
        """
        dropout_prob = self.args.dropout_prob
        first_hidden_dim = self.args.fp_dim
        hidden_dim = self.args.hidden_dim
        num_layers = self.args.num_layers
        num_classes = self.args.num_classes

        dropout = Dropout(dropout_prob)
        activation_fn = ReLU()

        if num_layers == 1:
            network = [dropout, Linear(first_hidden_dim, num_classes)]
        else:
            network = [dropout, Linear(first_hidden_dim, hidden_dim)]
            for i in range(num_layers - 2):
                network.extend([activation_fn, dropout, Linear(hidden_dim, hidden_dim)])
            network.extend([activation_fn, dropout, Linear(hidden_dim, num_classes)])
        self.feed_forward = Sequential(*network)

    def forward(self, data):
        """
        Runs the PFPNetwork on the input
        :param input: batch of Proximity Graphs
        :return: Output of the PFPNetwork
        """
        out = self.feed_forward(self.encoder(data.x))

        return out.view(-1)