from torch.utils.data import Subset
import torch

from sklearn.model_selection import KFold
import numpy as np

import os.path as osp
import os

from pgn.data.data_utils import normalize_targets, normalize_distance
from pgn.train.train_model import train_model


def cross_validation(args, train_data):
    """
    Function to run cross validation to train a model.
    :param args: TrainArgs object containing the parameters for the cross-validation run.
    :param train_data: The training to be used in cross-validation
    :param test_data: The testing data if loaded to be used to evaluate model performance.
    :return:
    """
    folds = args.cv_folds
    base_dir = args.save_dir
    seed = args.seed
    norm_targets = args.normalize_targets
    norm_dist = args.normalize_dist

    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    fold = 0
    evals = []
    for train_index, valid_index in kfold.split(train_data):

        with torch.no_grad():
        # Normalize targets and dist
            if norm_targets:
                label_mean = train_data.data.y[list(train_index)].mean()
                label_std = train_data.data.y[list(train_index)].std()
                args.label_mean, args.label_std = label_mean, label_std
                train_data.data.y = (train_data.data.y - label_mean) / label_std

            if norm_dist:
                dist_mean = train_data.data.edge_attr[list(train_index), 0].mean()
                dist_std = train_data.data.edge_attr[list(train_index), 0].std()
                args.distance_mean, args.distance_std = dist_mean, dist_std
                train_data.data.edge_attr[:, 0] = (train_data.data.edge_attr[:, 0] - dist_mean) / dist_std

            # Split datasets
            fold_train = train_data[list(train_index)]
            fold_valid = train_data[list(valid_index)]

        fold_dir = osp.join(base_dir, 'cv_fold_{0}'.format(fold))
        args.save_dir = fold_dir
        os.mkdir(fold_dir)

        # Run training
        model, eval = train_model(args, fold_train, fold_valid)
        evals.append(eval)

        # Revert train_dataset std and mean for next CV round
        if norm_targets:
            train_data.data.y = (train_data.data.y * label_std) + label_mean

        if norm_dist:
            train_data.data.edge_attr[:, 0] = (train_data.data.edge_attr[:, 0] * dist_std) + dist_mean

        fold += 1
        model = model.to('cpu')
        del model
        torch.cuda.empty_cache()


    return None, evals
