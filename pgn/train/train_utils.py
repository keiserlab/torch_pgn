from pgn.data.dmpnn_utils import BatchProxGraph
import torch
import torch.nn.functional as F

import os.path as osp
import os

def _format_batch(train_args, data):
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
    loss_function = args.loss_fucntion
    return lambda predicted, actual, num_grahps: loss_possibilities[loss_function](predicted, actual, num_grahps)


def make_save_directories(save_directory):
    """
    Formats the empty save directory in order to have the proper format.
    :param save_directory: An empty directory where the output of training will be saved.
    :return: None
    """
    if len(os.listdir(save_directory)) != 0:
        raise ValueError(
            "The save directory is not empty. Please either clean the directory or specify and empty directory."
        )
    if not os.path.isdir(save_directory):
        raise ValueError(
            "The specified save directory does not exist. Please create an empty directory with this path or specify"
            "a different path."
        )
    # Save the pytorch model data and the arguments json to this directory
    model_dir = osp.join(save_directory, "model")
    # The results of the training: i.e. data splits, predicted vs. actual for specified data sets, any plots specified etc.
    results_dir = osp.join(save_directory, "results")
    os.mkdir(model_dir)
    os.mkdir(results_dir)