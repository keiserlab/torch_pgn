import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from torch_pgn.train.train_utils import predict


def plot_correlation(model, args, data_loader, mean=0, std=1, metrics=None, filename='train_correlation', fit=True):
    """
    Simple method to plot correlations for a model.
    :param model: The model to be evaluated
    :param args: TrainArgs type object containing the parameters used to train the model
    :param data_loader: The torch dataloader object containing the working_data to be plotted.
    :param filename: The name of the plot file.
    :param fit: Boolean toggle for whether to include a trendline on the plot.
    :return: None (saved plot to savedir results file).
    """
    predictions, labels = predict(model, data_loader, args, return_labels=True, remove_norm=True)
    save_file = osp.join(args.save_dir, 'results', filename)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(labels, predictions, alpha=0.2)
    if fit:
        z = np.polyfit(labels, predictions, 1)
        p = np.poly1d(z)
        ax.set_title(filename)
        ax.plot(predictions, p(predictions), 'r--')
    if metrics is not None and metrics.get('r2') is not None and metrics.get('pcc') is not None:
        text = f"$R^2 = {metrics.get('r2'):0.3f}$\n$PCC = {metrics.get('pcc'):0.3f}$"
        plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    ax.set(xlabel='Ground Truth', ylabel='Model Prediction')
    plt.savefig(save_file)
    plt.close(fig)