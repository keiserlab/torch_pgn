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

    nn_conv_dropout_prob: float = 0.5
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

    ff_dim_1: int = None
    """dimension of the first ff layer after the fingerprint. If not manually set this will be set to hidden dimension."""

    num_layers: int = 3
    """Number of FC layers"""

    num_classes: int = 1
    """Number of classes being predicted by the model"""

    def process_args(self) -> None:
        if self.ff_dim_1 is None:
            self.ff_dim_1 = self.hidden_dim

class DataArgs(Tap):
    """
    Class used to store the arguments used to construct and format the dataset being used
    """
    #### Probably will end up going into a shared arguments class, but for now keep as is. ####
    seed: int = 0
    """Random seed used for pytorch in order to ensure reproducibility"""

    num_workers: int = 8
    """Number of processing units to use in working_data processing and loading."""

    #### Begin working_data arguments for loading the raw pdb and molecule working_data and processing them into ProximityGraphs. ####
    save_graphs: bool = True
    """Boolean toggle of whether to save the proximity graphs to harddisk. False will keep the entire processing in
    ram. Not saving the graphs is only advised if you are sure you will not be using the graphs again as this step takes
    a decent amount of time depending upon the dataset size and complexity of the proximity graph."""

    directed: bool = True
    """Boolean toggle for whether to make the proximity graph undirected. Only valid for use with dmpnn."""
    #TODO: Figure out how to make certain arguments required conditional upon other requirements.

    raw_data_path: str = None
    """The path to the raw working_data for Many vs. Many datasets. See description of formatting requirements in documentation."""

    raw_pdb_path: str = None
    """Path to the receptor pdb file used to construct all Proximity Graphs in OneVsMany datasets."""

    raw_mol_path: str = None
    """Path to the mol file used to construct all Proximity Graphs in OneVsMany datasets."""

    dataset_type: Literal['one_v_many', 'many_v_many', 'fp']
    """The type of dataset being loaded. The choice are one_v_many, which is when you have many ligands bound to a 
    single receptor. The other type of supported dataset is for many receptor ligand pairs. See dataset object
    documentation for further details.
    """

    label_file: str = None
    """"The path to the file containing the labels."""

    label_col: int = 3
    """The column number to use to fetch the labels."""

    #### Begin working_data arguments for loading proximity graphs into pytorch dataloader ####
    data_path: str
    """The path to place the formatted proximity graphs for input into pytorch Dataset (ProximityGraphDataset)."""

    transforms: List[str] = ['one_hot']
    """Transforms to apply to the dataset."""

    split_type: Literal['random', 'defined', 'defined_test'] = 'random'
    """The mode used to split the working_data into train, validation and test. Random will randomly split the working_data. Defined
    will use a defined split based on the name of each graph. Defined_test will do the same, but only for loading
    defined train and test splits (validation will be randomly picked from train)."""

    split_dir: str = None
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
    """Whether to process raw working_data or just look in the data_path for formated proximity graphs"""

    validation_percent: int = 0.10
    """The percentage of the training working_data to put into the validation set in the case of a random split."""

    test_percent: int = 0.10
    """The percentage of the training working_data to hold out into the test set in the case of a random split."""

    label_mean: float = 0.
    """Mean correction used for target normalization."""

    label_std: float = 1.
    """Std deviation correction used for target normalization."""

    distance_mean: float = 0.
    """Mean correction used for distance normalization."""

    distance_std: float = 1.
    """Std deviation correction used for distance normalization."""

    save_splits: bool = False
    """Write the training/validation/test splits to the """

    train_index: List = None
    """Stores the index of the training set in order to ensure no dataset contamination."""

    fp_format: Literal['sparse', 'dense'] = 'sparse'
    """The format of the fingerprints in the raw_data directory if the dataset type == fp."""

    fp_column: str = 'fp'
    """The name of the column that contains the raw_fp data for loading with fp dataset type."""

    fp_dim: int = 4096
    """The dimension of the output feature vector from the encoder for either pfp or fp dataset types."""

    ###################### Dataloader Args #############################################################################
    enable_molgraph: bool = True
    """Include molgraph attribute in dataset to enable use with dmpnn"""

    enable_interacting_mask: bool = True
    """Include molgraph attribute in dataset to enable use with dmpnn"""



    def process_args(self):
        if self.raw_data_path is not None and self.label_file is None:
            self.label_file = osp.join(self.raw_data_path, 'index', '2016_index.lst')


class TrainArgs(DataArgs, FFArgs, EncoderArgs):
    """Class used to store the model independent arguments used for training the NNs"""
    ############ Required arguments ###################################################################################

    save_dir: str
    """The directory to save the checkpoint files and model outputs."""

    ############ Optional arguments ###################################################################################
    device: str = 'cpu'
    """The device to train then model on/location of working_data and pytorch model. e.g. 'cpu' or 'cuda'"""
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
    decay_delay: int = 20
    """Number of epoch to delay before starting weight decay."""
    epochs: int = 50
    """The number of training epochs to run."""
    metrics: List[str] = ['rmse', 'mse', 'pcc', 'r2']
    """The metrics used to evaluate the validation and if desired test performance of the model. Valid choices currently
    include: rmse, mse, pcc, r2."""
    plot_correlations: bool = True
    """Boolean toggle of whether to plot the correlations for train, validation and test (if loaded)."""
    cross_validate: bool = False
    """Boolean toggle of whether to run cross-validation. Pairs cv_folds argument. If cv_folds is set then cross_validate
    will be set to True."""
    cv_folds: int = None
    """Number of folds to use in cross-validation. If this is set cross-validations will be automatically used."""
    tensorboard_logging: bool = True
    """Whether or not to track model training with tensorboard."""
    mode: Literal['experiment', 'evaluate'] = 'experiment'

    def process_args(self):
        if self.ff_dim_1 is None:
            self.ff_dim_1 = self.hidden_dim
        if self.raw_data_path is not None and self.label_file is None:
            self.label_file = osp.join(self.raw_data_path, 'index', '2016_index.lst')
        if self.cv_folds is not None:
            self.cross_validate = True
        if self.encoder_type == 'dmpnn' and 'molgraph' not in self.transforms :
            self.transforms.append('molgraph')
        if self.encoder_type == 'ggnet':
            self.fp_dim = self.nn_conv_in_dim * 2
        #Depricate seperate dropout probability for nn_conv and ff
        #self.nn_conv_dropout_prob = self.dropout_prob
        if self.dataset_type == 'fp':
            self.normalize_dist = False
            self.encoder_type = 'fp'
        if self.mode == 'evaluate':
            self.load_test = True


class HyperoptArgs(TrainArgs):
    num_iters: int = 10
    """The number of iterations of model optimization to be run"""
    minimize_score: bool = True
    """Whether the score is minimized or maximized during hyperparameter optimization."""
    search_keys: List = ['fp_dim', 'ffn_num_layers', 'dropout', 'ffn_hidden_size']





