from pgn.data.dmpnn_utils import BatchProxGraph
from pgn.models.model import PFPNetwork
from pgn.args import TrainArgs

from argparse import Namespace

import torch
import torch.nn.functional as F

from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from tqdm import tqdm
import numpy as np

import os.path as osp
import os

def format_batch(train_args, data):
    if train_args.encoder_type == 'd-mpnn':
        return BatchProxGraph(data)
    else:
        return data


def rmse_loss(predicted, actual, num_graphs):
    """
    Returns the RMSE loss for given predicted and ground_truth values for a given number of graphs (batch size)
    :param predicted: The output of the model on the given batch of data
    :param actual: The ground truth value for the given graphs
    :param num_graphs: The number of graphs being evaluated
    :return: The average RMSE loss over the given graphs.
    """
    return torch.sqrt(torch.sum((predicted - actual) ** 2) / num_graphs)

def mse_loss(predicted, actual, num_graphs):
    """
    Returns the MSE loss for given predicted and ground_truth values for a given number of graphs (batch size)
    :param predicted: The output of the model on the given batch of data
    :param actual: The ground truth value for the given graphs
    :param num_graphs: The number of graphs being evaluated (not used)
    :return: The average MSE loss over the given graphs.
    """
    return F.mse_loss(predicted, actual, reduction="mean")


def parse_loss(args):
    """
    Parses the arg for loss function and returns the appropriate loss function (currently either RMSE or MSE).
    :param args: An instance of train args
    :return: The loss function
    """
    loss_possibilities = {'rmse': rmse_loss, 'mse': mse_loss}
    loss_function = args.loss_function
    return lambda predicted, actual, num_grahps: loss_possibilities[loss_function](predicted, actual, num_grahps)


def make_save_directories(save_directory):
    """
    Formats the empty save directory in order to have the proper format.
    :param save_directory: An empty directory where the output of training will be saved.
    :return: None
    """
    if len([directory for directory in os.listdir(save_directory) if not directory.startswith('.')]) != 0:
        raise ValueError(
            "The save directory (save_dir) is not empty. "
            "Please either clean the directory or specify and empty directory."
        )
    if not os.path.isdir(save_directory):
        raise ValueError(
            "The specified save directory (save_dir) does not exist. Please create an empty directory with this "
            "path or specify a different path."
        )
    # Save the pytorch model data and the arguments json to this directory
    model_dir = osp.join(save_directory, "model")
    # The results of the training: i.e. data splits, predicted vs. actual for specified data sets, any plots specified etc.
    results_dir = osp.join(save_directory, "results")
    os.mkdir(model_dir)
    os.mkdir(results_dir)


def predict(model, data_loader, args, progress_bar=True):
    """
    Return the result when the specified model is applied to the data in the data_loader
    :param model: The model being used to predict
    :param data_loader: The pytorch_geometric dataloader object containing the data to be evaluated.
    :param args: The TrainArgs object that contains the required accessory arguments
    :return: The raw output of the model as a numpy array.
    """
    model.eval()
    preds = []
    for data in data_loader:
        data = data.to(args.device)
        preds.append(model(format_batch(args, data)).cpu().detach().numpy())
    preds = np.hstack(preds)
    return preds


def get_labels(data_loader):
    """
    Helper function to get the ground truth values
    :param data_loader: dataloader to be get the ground truth values from
    :return: A labels array (np)
    """
    labels = []
    for data in data_loader:
        labels.append(data.y.cpu().detach().numpy())
    labels = np.hstack(labels)
    return labels


def get_metric_functions(metrics):
    """
    Returns the relevant metric functions given a list of valid metrics
    :param metrics: A list of valid metrics in {'rmse', 'mse', 'r2', 'pcc', 'aucroc', 'aucprc'}
    :return: A dictionary that maps metrics to functions.
    """
    metric_map = {}
    for metric in metrics:
        if metric == 'rmse':
            metric_map['rmse'] = lambda truth, predicted: np.sqrt(np.sum((predicted - truth) ** 2) / truth.shape[0])
        elif metric == 'mse':
            metric_map['mse'] = lambda truth, predicted: np.sum((predicted - truth) ** 2) / truth.shape[0]
        elif metric == 'r2':
            metric_map['r2'] = lambda truth, predicted: r2_score(truth, predicted)
        elif metric == 'pcc':
            metric_map['pcc'] = lambda truth, predicted: pearsonr(truth, predicted)[0]
    return metric_map


def save_checkpoint(path, model, args):
    """
    Save the current state of training including the model and the arguments used to instantiate the model.
    :param path: The path to save the state to.
    :param model: The current model.
    :param args: The training arguments used to construct/parameterize the model.
    :return: None
    """
    if args is not None:
        args = Namespace(**args.as_dict())
    state = {
        'args': args,
        'state_dict': model.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(path, device):
    """
    Loads a checkpoint.
    :param path: The path which contains the checkpoint file.
    :param device: The device to loader the model to.
    :return: The loaded model file.
    """

    state = torch.load(path, map_location=lambda storage, loc: storage)
    args = TrainArgs()
    args.from_dict(vars(state['args']), skip_unsettable=True)
    model_state_dict = state['state_dict']

    if device is not None:
        args.device = device

    model = PFPNetwork(args, args.node_dim, args.edge_dim)
    model.load_state_dict(model_state_dict)

    return model