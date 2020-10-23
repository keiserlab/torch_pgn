"""Classes that pass arguements to networks for training and easy saving and retrieval"""

from tap import Tap
from typing_extensions import Literal


class EncoderArgs(Tap):
    """Class used to store the arguments used for input to the message passing NN encoder."""


    nn_conv_in_dim: int = 16
    """In dimension to the NN_conv layer"""
    nn_conv_internal_dim: int = 128
    """Dimension of the internal ff_network in the NN_conv layer"""
    nn_conv_out_dim: int = 16
    """output dimension of the nn_conv output matrix (nn_conv_out_dim x nn_conv_out_dim)"""
    nn_conv_aggr: Literal['add', 'mean', 'max'] = 'mean'
    """type of pool used in the nn_conv layer"""
    pool_type: Literal['add', 'mean', 'max'] = 'add'
    """pooling applied as part of readout function"""
    sparse_type: Literal['log_softmax', 'softmax', 'none'] = 'log_softmax'
    """sparsification function used before final readout"""
    fp_norm: int = True
    """whether or not to apply batch norm to the fp_layer before readout"""
    fp_dim: int = 4096
    """The dimension of the output feature vector"""
    depth: int = 3
    """The number of message passing steps"""
    skip_connections: bool = True
    """Toggle for whether to readout the fp vector at only time=T (False) or to aggregate all the feature
        vectors by adding the readout after each message passing step to the final readout vector."""

class FFArgs(Tap):
    """Class used to store the arguments used for input to the readout network."""
    dropout_prob: float = 0.0
    """probability of dropout applied between hidden layers"""
    hidden_dim: int = 400
    """size of the FC layers"""
    num_layers: int = 4
    """Number of FC layers"""
    num_classes: int = 1
    """Number of classes being predicted by the model"""

class TrainArgs(Tap):
    """Class used to store the model independent arguments used for training the NNs"""
    pass


class DataArgs(Tap):
    """
    Class used to store the arguments used to construct and format the dataset being used
    """
    #### Probably will end up going into a shared arguments class, but for now keep as is####
    seed: int = 0
    """Random seed used for pytorch in order to ensure reproducibility"""


class AggregatedArgs(Tap):
    """Wrapper class used to store the encoder, ff and train args classes"""
    def configure(self):
        self.add_subparsers(help='sub-command help')
        self.add_subparser('encoder_args', EncoderArgs, help='encoder help')
        self.add_subparser('FF_args', FFArgs, help='FF help')