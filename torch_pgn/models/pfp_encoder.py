import torch
import torch.nn.functional as F
from torch.nn import ReLU, Sequential, Linear, BatchNorm1d, Dropout

from torch_geometric.nn import NNConv

from torch_pgn.args import TrainArgs
from torch_pgn.models.nn_utils import get_sparse_fnctn, get_pool_function

class PFPEncoder(torch.nn.Module):
    """Message passing network used to encode interaction graphs for use in either regression or classification
    tasks."""
    def __init__(self, args: TrainArgs, node_dim: int, bond_dim: int):
        super(PFPEncoder, self).__init__()
        self.node_dim = node_dim
        self.bond_dim = bond_dim
        self.device = args.device
        self.nn_conv_in_dim = args.nn_conv_in_dim
        self.nn_conv_internal_dim = args.nn_conv_internal_dim
        self.nn_conv_out_dim = args.nn_conv_out_dim
        self.nn_conv_dropout_prob = args.nn_conv_dropout_prob
        self.nn_conv_aggr = args.nn_conv_aggr
        self.pool_type = args.pool_type
        self.sparse_type = args.sparse_type
        self.fp_norm = args.fp_norm
        self.fp_dim = args.fp_dim
        self.depth = args.depth
        self.skip_connections = args.skip_connections
        self.ligand_only_readout = args.ligand_only_readout
        self.split_conv = args.split_conv
        self.covalent_only_depth = args.covalent_only_depth
        self.one_step_convolution = args.one_step_convolution

        # define model layers
        self.construct_nn_conv()

        self.atom_expand_nn = Sequential(Linear(self.node_dim, self.nn_conv_in_dim), ReLU())
        self.pool = get_pool_function(self.pool_type)
        self.sparsify = get_sparse_fnctn(self.sparse_type)
        self.expand_to_fp_nn = Sequential(Linear(self.nn_conv_in_dim, self.fp_dim), ReLU())

        # define batch normalization
        if self.fp_norm:
            self.norm_layer = BatchNorm1d(num_features=self.fp_dim)

    def construct_nn_conv(self):
        #Construct the NN Conv network
        if not self.split_conv:
            feed_forward = Sequential(Linear(self.bond_dim, self.nn_conv_internal_dim),
                                      ReLU(),
                                      Dropout(p=self.nn_conv_dropout_prob),
                                      Linear(self.nn_conv_internal_dim, self.nn_conv_out_dim))
            self.conv = NNConv(self.nn_conv_in_dim, self.nn_conv_in_dim, feed_forward, aggr=self.nn_conv_aggr)
        else:
            feed_forward_covalent = Sequential(Linear(self.bond_dim, self.nn_conv_internal_dim),
                                      ReLU(),
                                      Dropout(p=self.nn_conv_dropout_prob),
                                      Linear(self.nn_conv_internal_dim, self.nn_conv_out_dim))
            feed_forward_spacial = Sequential(Linear(self.bond_dim, self.nn_conv_internal_dim),
                                      ReLU(),
                                      Dropout(p=self.nn_conv_dropout_prob),
                                      Linear(self.nn_conv_internal_dim, self.nn_conv_out_dim))
            self.conv_covalent = NNConv(self.nn_conv_in_dim, self.nn_conv_in_dim, feed_forward_covalent, aggr=self.nn_conv_aggr)
            self.conv_spacial = NNConv(self.nn_conv_in_dim, self.nn_conv_in_dim, feed_forward_spacial, aggr=self.nn_conv_aggr)


    def forward(self, data):
        # Define output vector
        fingerprint = torch.zeros(self.fp_dim, dtype=torch.float32, requires_grad=True, device=self.device)
        # Expand atom featurization for input to NNConv
        message = self.atom_expand_nn(data.x)
        # Begin message passing steps
        for i in range(self.depth):
            message = self._apply_convolution(message, data, i)
            # expand message for incorporation into final feature vector
            expanded_message = F.relu(self.expand_to_fp_nn(message.unsqueeze(0)))
            # readout for depth i
            if self.ligand_only_readout:
                ligand_message_mask = data.x[:, -1] == 1.
                expanded_message = expanded_message.squeeze()[ligand_message_mask].unsqueeze(0)
                ligand_batch = data.batch[ligand_message_mask]
                intermediate = self.pool(expanded_message, ligand_batch)
            else:
                intermediate = self.pool(expanded_message, data.batch)

            readout = self.sparsify(intermediate)

            # skip connections if specified or final message passing step to readout vector
            if self.skip_connections or i == self.depth - 1:
                fingerprint = fingerprint.to(readout.device) + readout
        fingerprint = fingerprint.squeeze()
        if self.fp_norm:
            return self.norm_layer(fingerprint)
        else:
            return fingerprint

    def _apply_convolution(self, message, data, depth):
        """In the case where covalent and spacial bonds have different message passing functions this function applies
        the relevant message passing for the given bond types."""
        # Should probably reimplement this with a custon nnconv layer in order to allow for other aggregation schemes
        # aside from the add hardcoded here.
        if self.split_conv:
            conv_covalent, conv_spacial = self.conv_covalent, self.conv_spacial
        else:
            conv_covalent, conv_spacial = self.conv, self.conv
        if depth < self.covalent_only_depth:
            covalent_mask = data.edge_attr[:,-1] == 0
            covalent_output = conv_covalent(message, data.edge_index[:, covalent_mask],
                                                 data.edge_attr[covalent_mask, :])
            message = F.relu(covalent_output)
        else:
            # In the case of one-step convolution conv_covalent == conv_spacial.
            covalent_mask = data.edge_attr[:, -1] == 0
            spacial_mask = data.edge_attr[:, -1] == 1
            covalent_output = conv_covalent(message, data.edge_index[:, covalent_mask], data.edge_attr[covalent_mask, :])
            spacial_output = conv_spacial(message, data.edge_index[:, spacial_mask], data.edge_attr[spacial_mask, :])
            message = F.relu(covalent_output + spacial_output)
        return message
