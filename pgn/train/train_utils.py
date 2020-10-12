from pgn.data.dmpnn_utils import BatchProxGraph
import torch
import torch.nn.functional as F

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