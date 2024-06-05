from torch.utils.data import Subset
import torch

from sklearn.model_selection import KFold
import numpy as np

from copy import deepcopy

import os.path as osp
import os

from torch_pgn.data.data_utils import normalize_targets, normalize_distance
from torch_pgn.train.train_model import train_model


def cross_validation(args, train_data):
    """
    Function to run cross validation to train a model.
    :param args: TrainArgs object containing the parameters for the cross-validation run.
    :param train_data: The training to be used in cross-validation
    :param test_data: The testing working_data if loaded to be used to evaluate model performance.
    :return:
    """
    folds = args.cv_folds
    base_dir = args.save_dir
    seed = args.seed
    norm_targets = args.normalize_targets
    norm_dist = args.normalize_dist and args.include_dist
    train_index = args.train_index
    # Ensure the test set is not loaded.
    args.load_test = False

    fold = 0
    evals = []
    while fold < folds:

        fold_train_index, fold_valid_index = _split_dataset(train_index, fold, folds)

        with torch.no_grad():
        # Normalize targets and dist
            if norm_targets:
                train_data.data.y, label_stats = normalize_targets(train_data, index=fold_train_index)
                args.label_mean, args.label_std = label_stats

            if norm_dist:
                train_data.data.edge_attr, dist_stats = normalize_distance(train_data, args=args,
                                                                              index=fold_train_index)
                args.distance_mean, args.distance_std = dist_stats

            # Split datasets
            fold_train = train_data[fold_train_index]
            fold_valid = train_data[fold_valid_index]

        fold_dir = osp.join(base_dir, 'cv_fold_{0}'.format(fold))
        args.save_dir = fold_dir
        os.mkdir(fold_dir)

        # Run training
        model, eval = train_model(args, fold_train, fold_valid)
        evals.append(eval)

        fold += 1
        model = model.to('cpu')
        del model
        torch.cuda.empty_cache()

        if norm_targets:
            train_data.data.y = (train_data.data.y * args.label_std) + args.label_mean

        if norm_dist:
            train_data.data.edge_attr[:, 0] = (train_data.data.edge_attr[:, 0] * args.distance_std) + args.distance_mean

    return None, evals


def _split_dataset(index, fold, num_folds):
    """
    Splits the dataset into validation and train for each fold.
    :param index: The index to be split.
    :param fold: The validation fold currently on.
    :param num_folds: The total number of cross-validation folds.
    :return: The train_index and validation_index
    """
    num_examples = len(index)
    valid_percent = 1 / num_folds
    valid_size = int(valid_percent * num_examples)
    valid_begin, valid_end = valid_size * fold, valid_size * fold + valid_size
    valid_index = index[valid_begin:valid_end]
    train_index = deepcopy(index)
    del train_index[valid_begin:valid_end]
    return train_index, valid_index