from pgn.train.train_utils import predict, get_metric_functions, get_labels

def evaluate(model, data_loader, args, metrics, mean=0, std=1):
    """
    Function used to evaluate the model performance on a given dataset.
    :param model: The model to be evaluated.
    :param data_loader: The dataloader containing the data the model with be evaluated on.
    :param args: TrainArgs object containing the relevant arguments for evaluation.
    :param metric: The metrics used to evaluate the model on the given data.
    :param mean:The mean of the non-normalized data.
    :param std: The stddev. of the non-normalized data.
    :return: The value of the metrics.
    """
    metric_function_map = get_metric_functions(metrics)

    predictions = predict(model=model,
                          data_loader=data_loader,
                          args=args)

    labels = get_labels(data_loader=data_loader)

    results = {}

    for metric in metric_function_map.keys():
        results[metric] = metric_function_map[metric](predictions, labels)

    return results