import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from pgn.train.train_utils import predict

def plot_correlation(model, args, data_loader, filename='train_correlation', fit=True):
    """
    Simple method to plot correlations for a model.
    :param model: The model to be evaluated
    :param args: TrainArgs type object containing the parameters used to train the model
    :param data_loader: The torch dataloader object containing the data to be plotted.
    :param filename: The name of the plot file.
    :param fit: Boolean toggle for whether to include a trendline on the plot.
    :return: None (saved plot to savedir results file).
    """
    predictions, labels = predict(model, data_loader, args, return_labels=True)
    save_file = osp.join(args.save_dir, 'results', filename)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(labels, predictions, alpha=0.2)
    if fit:
        z = np.polyfit(labels, predictions, 1)
        p = np.poly1d(z)
        ax.set_title(filename)
        ax.plot(predictions, p(predictions), 'r--')
    ax.set(xlabel='Ground Truth', ylabel='Model Prediction')
    plt.savefig(save_file)