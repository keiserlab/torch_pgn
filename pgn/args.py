"""Classes that pass arguements to networks for training and easy saving and retrieval"""

from tap import Tap
from typing_extensions import Literal
from typing import List

import os.path as osp


class EncoderArgs(Tap):
    """Class used to store the arguments used for input to the message passing NN encoder."""
    nn_conv_in_dim: int = 16
    """In dimension to the NN_conv layer"""

    nn_conv_internal_dim: int = 128
    """Dimension of the internal ff_network in the NN_conv layer"""

    nn_conv_out_dim: int = 16*16
    """output dimension of the nn_conv output matrix (nn_conv_out_dim x nn_conv_out_dim)"""

    nn_conv_dropout: float = 0.5
    """dropout probability used in the nn_conv ff_network"""

    nn_conv_aggr: Literal['add', 'mean', 'max'] = 'mean'
    """type of pool used in the nn_conv layer"""

    pool_type: Literal['add', 'mean', 'max'] = 'add'
    """pooling applied as part of readout function"""

    sparse_type: Literal['log_softmax', 'softmax', 'none'] = 'log_softmax'
    """sparsification function used before final readout"""

    fp_norm: bool = True
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

class DataArgs(Tap):
    """
    Class used to store the arguments used to construct and format the dataset being used
    """
    #### Probably will end up going into a shared arguments class, but for now keep as is. ####
    seed: int = 0
    """Random seed used for pytorch in order to ensure reproducibility"""

    num_workers: int = 8
    """Number of processing units to use in data processing and loading."""

    #### Begin data arguments for loading the raw pdb and molecule data and processing them into ProximityGraphs. ####
    save_graphs: bool = True
    """Boolean toggle of whether to save the proximity graphs to harddisk. False will keep the entire processing in
    ram. Not saving the graphs is only advised if you are sure you will not be using the graphs again as this step takes
    a decent amount of time depending upon the dataset size and complexity of the proximity graph."""

    directed: bool = False
    """Boolean toggle for whether to make the proximity graph undirected"""
    #TODO: Figure out how to make certain arguments required conditional upon other requirements.

    raw_data_path: str = None
    """The path to the raw data for Many vs. Many datasets. See description of formatting requirements in documentation."""

    raw_pdb_path: str = None
    """Path to the receptor pdb file used to construct all Proximity Graphs in OneVsMany datasets."""

    raw_mol_path: str = None
    """Path to the mol file used to construct all Proximity Graphs in OneVsMany datasets."""

    dataset_type: Literal['one_v_many', 'many_v_many']
    """The type of dataset being loaded. The choice are one_v_many, which is when you have many ligands bound to a 
    single receptor. The other type of supported dataset is for many receptor ligand pairs. See dataset object
    documentation for further details.
    """

    label_file: str = None
    """"The path to the file containing the labels."""

    label_col: int = 3
    """The column number to use to fetch the labels."""

    #### Begin data arguments for loading proximity graphs into pytorch dataloader ####
    data_path: str
    """The path to place the formatted proximity graphs for input into pytorch Dataset (ProximityGraphDataset)."""

    transforms: Literal['one_hot', 'molgraph', 'none'] = ['one_hot'] # TODO:Fix me
    """Transforms to apply to the dataset."""

    split_type: Literal['random', 'defined']
    """The mode used to split the data into train, validation and test. Random will randomly split the data. Discrete
    will use a defined split based on the name of each graph."""

    split_location: str = None
    """If the split type is defined, this is the location of the directory with files train.txt, validation.txt and 
    test.txt (see documentation for the format of these files)."""
    normalize_targets: bool = True
    """"Boolean toggle of whether to normalize the targets to have mean of 0 and stddev of 1."""

    include_dist: bool = True
    """Boolean toggle of whether to include distance information in the bond features of the proximity graphs."""

    normalize_dist: bool = True
    """Boolean toggle of whether to normalize the distance to have mean 0 and stddev of 1."""

    load_test: bool = True
    """Boolean toggle of whether or not to load the test set for evaluation."""

    construct_graphs: bool = True
    """Whether to process raw data or just look in the data_path for formated proximity graphs"""

    validation_percent: int = 0.20
    """The percentage of the training data to put into the validation set in the case of a random split."""

    test_percent: int = 0.20
    """The percentage of the training data to hold out into the test set in the case of a random split."""

    label_mean: float = 0.
    """Mean correction used for target normalization."""

    label_std: float = 1.
    """Std deviation correction used for target normalization."""

    distance_mean: float = 0.
    """Mean correction used for distance normalization."""

    distance_std: float = 1.
    """Std deviation correction used for distance normalization."""

    def process_args(self):
        print('here')
        if self.raw_data_path is not None and self.label_file is None:
            self.label_file = osp.join(self.raw_data_path, 'index', '2016_index.lst')


class TrainArgs(DataArgs, FFArgs, EncoderArgs):
    """Class used to store the model independent arguments used for training the NNs"""
    ############ Required arguments ###################################################################################

    save_dir: str
    """The directory to save the checkpoint files and model outputs."""

    ############ Optional arguments ###################################################################################
    device: str = 'cpu'
    """The device to train then model on/location of data and pytorch model. e.g. 'cpu' or 'cuda'"""
    node_dim: int = None
    """The node feature size. Set during dataloading procedure."""
    edge_dim: int = None
    """The edge feature size. Set during dataloading procedure."""
    loss_function: str = 'mse'
    """The function used to evaluate the model. The default is mse. Valid options are: mse, rmse"""
    encoder_type: str = 'pfp'
    """Selects the encoder to be used, defaults to pfp network. Valid options are: pfp, d-mpnn"""
    torch_seed: int = 42
    """Pytorch dataloader seed."""
    seed: int = 0
    """Pytorch seed."""
    num_workers: int = 8
    """The number of works used in dataloading and batch generation."""
    batch_size: int = 256
    """The batch size used in training."""
    load_test: bool = False
    """Boolean toggle that indicates whether the test set should be loaded and evaluated."""
    lr: float = 3e-4
    """The initial learning rate used to train the model."""
    weight_decay: bool = False
    """Boolean toggle to indicate whether weight decay should be used during training"""
    epochs: int = 50
    """The number of training epochs to run."""
    metrics: List[str] = ['rmse', 'mse', 'pcc', 'r2']
    """The metrics used to evaluate the validation and if desired test performance of the model. Valid choices currently
    include: rmse, mse, pcc, r2."""
    plot_correlations: bool = True
    """Boolean toggle of whether to plot the correlations for train, validation and test (if loaded)."""



class HyperoptArgs(TrainArgs):
    num_iters: int = 10
    """The number of iterations of model optimization to be run"""
    minimize_score: bool = True
    """Whether the score is minimized or maximized during hyperparameter optimization."""





