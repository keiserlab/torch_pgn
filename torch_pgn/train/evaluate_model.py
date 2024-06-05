from torch_pgn.train.train_utils import predict

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import numpy as np

def evaluate(model, data_loader, args, metrics, mean=0, std=1, remove_norm=True):
    """
    Function used to evaluate the model performance on a given dataset.
    :param model: The model to be evaluated.
    :param data_loader: The dataloader containing the working_data the model with be evaluated on.
    :param args: TrainArgs object containing the relevant arguments for evaluation.
    :param metric: The metrics used to evaluate the model on the given working_data.
    :param mean:The mean of the non-normalized working_data.
    :param std: The stddev. of the non-normalized working_data.
    :return: The value of the metrics.
    """
    predictions, labels = predict(model=model,
                          data_loader=data_loader,
                          args=args, return_labels=True, remove_norm=remove_norm)

    results = {}
    if 'rmse' in metrics:
        results['rmse'] = np.sqrt(np.sum((predictions - labels) ** 2) / labels.shape[0])
    if 'mse' in metrics:
        results['mse'] = np.sum((predictions - labels) ** 2) / labels.shape[0]
    if 'r2' in metrics:
        results['r2'] = r2_score(labels, predictions)
    if 'pcc' in metrics:
        results['pcc'] = pearsonr(predictions, labels)[0]

    return results