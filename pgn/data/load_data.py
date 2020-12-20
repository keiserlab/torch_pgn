from pgn.data.data_utils import (parse_transforms,
                                 normalize_targets, normalize_distance
                                 )
from pgn.data.ProximityGraphDataset import ProximityGraphDataset

import torch
import numpy as np
import os.path as osp
import os

# TODO: Add a little bit more though into how we want to do defined splits. Probably have an option for only having
#  a defined test set i.e. defined_test option in split type


def load_proximity_graphs(args):
    """
    #TODO
    :param args:
    :return:
    """
    data_path = args.data_path
    seed = args.seed
    # Add function to parse transforms into composed format for application
    transforms = parse_transforms(args.transforms)
    split_type = args.split_type
    norm_targets = args.normalize_targets
    include_dist = args.include_dist
    norm_dist = args.normalize_dist
    load_test = args.load_test
    cross_validate = args.cross_validate

    if cross_validate:
        return _load_data_cross_validation(args)

    torch.manual_seed(seed)

    if split_type == 'random':

        train_dataset = ProximityGraphDataset(data_path, include_dist=include_dist)
        train_dataset.data = transforms(train_dataset.data)

        num_examples = len(train_dataset.data.name)

        rand = np.random.RandomState(args.seed)
        permutations = rand.choice(num_examples, num_examples)

        valid_begin, valid_end = 0, int(args.validation_percent * num_examples)
        test_begin, test_end = valid_end, int(args.test_percent * num_examples) + valid_end
        train_begin, train_end = test_end, num_examples

        train_index = permutations[train_begin:train_end]
        valid_index = permutations[valid_begin:valid_end]
        test_index = permutations[test_begin:test_end]

        validation_dataset = train_dataset[list(valid_index)]
        test_dataset = train_dataset[list(test_index)]
        train_dataset = train_dataset[list(train_index)]

        if args.save_splits:
            _save_splits(args.save_dir, (train_dataset, train_index),
                         validation=(validation_dataset, valid_index),
                         test=(test_dataset, test_index))

        if norm_targets:
            train_dataset.data.y, target_stats = normalize_targets(train_dataset, yield_stats=True)
            args.label_mean, args.label_std = target_stats

        if norm_dist:
            train_dataset.data.edge_attr, dist_stats = normalize_distance(train_dataset, yield_stats=True)
            args.distance_mean, args.distance_std = dist_stats


    elif split_type == 'defined_test':
        # TODO: make sure error handling of no split dir happens somewhere
        split_dir = args.split_dir
        train_names, valid_names, test_names = _load_splits(split_dir)
        train_names = np.hstack((train_names, valid_names))

        print(train_names)
        train_dataset = ProximityGraphDataset(data_path,
                                              include_dist=include_dist)

        test_dataset = _split_data(train_dataset, test_names)
        train_dataset = _split_data(train_dataset, train_names)

        print(test_dataset, train_dataset)


    elif split_type == 'defined':
        #TODO: Figure out the best way for this to work
        valid_begin, valid_end = args.validation_splits
        train_begin, train_end = args.train_splits

        train_dataset = ProximityGraphDataset(data_path, include_dist=include_dist)
        train_dataset.data = transforms(train_dataset.data)

        validation_dataset = train_dataset[valid_begin:valid_end]
        train_dataset = train_dataset[train_begin:train_end]

        # TODO: Fix norms
        if norm_targets:
            train_dataset.data.y, target_stats = normalize_targets(train_dataset, yield_stats=True)

        if norm_dist:
            train_dataset.data.edge_attr, dist_stats = normalize_distance(train_dataset, yield_stats=True)

        if load_test:
            test_dataset = ProximityGraphDataset(data_path, include_dist=include_dist, mode='test')
            test_dataset.data = transforms(test_dataset.data)

            if norm_targets:
                test_dataset.data.y = normalize_targets(test_dataset, mean=target_stats[0],
                                                        std=target_stats[1], yield_stats=False)

            if norm_dist:
                test_dataset.data.edge_attr = normalize_distance(test_dataset, mean=dist_stats[0],
                                                        std=dist_stats[1], yield_stats=False)

    else:
        raise ValueError('Invalid dataset type. Please choose from <random> or <defined>')

    args.node_dim = train_dataset.data.x.numpy().shape[1]
    args.edge_dim = train_dataset.data.edge_attr.numpy().shape[1]

    if load_test:
        return train_dataset, validation_dataset, test_dataset

    else:
        return train_dataset, validation_dataset


def _load_data_cross_validation(args):
    """
    Helper function to keep main function from getting to complicated and clutter. Handles the loading and returning
    or data when cross-validation will be used to train and evaluate/select the model.
    :param args: The Argument object.
    :return: The datasets specified by args (either train alone if load_test is set to False or both train and test).
    """
    data_path = args.data_path
    seed = args.seed
    # Add function to parse transforms into composed format for application
    transforms = parse_transforms(args.transforms)
    split_type = args.split_type
    norm_targets = args.normalize_targets
    include_dist = args.include_dist
    norm_dist = args.normalize_dist
    load_test = args.load_test

    torch.manual_seed(seed)

    if split_type == 'random':

        train_dataset = ProximityGraphDataset(data_path, include_dist=include_dist)
        train_dataset.data = transforms(train_dataset.data)

        num_examples = len(train_dataset.data.name)
        # TODO: make this truly random by using shuffled indexing

        test_begin, test_end = 0, int(args.test_percent * num_examples)
        train_begin, train_end = test_end, num_examples

        test_dataset = train_dataset[test_begin: test_end]
        train_dataset = train_dataset[train_begin:train_end]

    elif split_type == 'defined':

        train_begin, train_end = args.train_splits

        train_dataset = ProximityGraphDataset(data_path, include_dist=include_dist)
        train_dataset.data = transforms(train_dataset.data)

        train_dataset = train_dataset[train_begin:train_end]

        if load_test:
            test_dataset = ProximityGraphDataset(data_path, include_dist=include_dist, mode='test')
            test_dataset.data = transforms(test_dataset.data)

    else:
        raise ValueError('Invalid dataset type. Please choose from <random> or <defined>')

    args.node_dim = train_dataset.data.x.numpy().shape[1]
    args.edge_dim = train_dataset.data.edge_attr.numpy().shape[1]

    if load_test:
        return train_dataset, test_dataset

    else:
        return train_dataset


def _save_splits(base_dir, train, validation, test):
    """
    Helper function to save the data splits.
    :param base_dir: The base_directory to write the splits directory containing the saved splits
    :param train: Tuple(train_dataset, train_index)
    :param validation: Tuple(validation_dataset, validation_index) .
    :param test: Tuple(test_dataset, test_index).
    """
    split_dir = osp.join(base_dir, 'splits')
    os.mkdir(split_dir)
    train_data, train_index = train
    np.save(osp.join(split_dir, 'train.npy'), np.array(train_data.data.name)[train_index])
    if validation is not None:
        val_data, val_index = validation
        np.save(osp.join(split_dir, 'validation.npy'), np.array(val_data.data.name)[val_index])
    if test is not None:
        test_data, test_index = test
        np.save(osp.join(split_dir, 'test.npy'), np.array(test_data.data.name)[test_index])


def _load_splits(split_dir):
    """
    Loads the splits from split_dir
    :param split_dir: Directory containing train.npy, validation.npy and test.npy
    :return: the name lists for each of the defined splits (train, valid, test).
    """
    train_path = osp.join(split_dir, 'train.npy')
    valid_path = osp.join(split_dir, 'validation.npy')
    test_path = osp.join(split_dir, 'test.npy')

    train_name = np.load(train_path)
    valid_name = np.load(valid_path)
    test_name = np.load(test_path)

    return train_name, valid_name, test_name


def _split_data(dataset, names):
    mask = np.vectorize(lambda value: value in names)(np.array(dataset.data.name))
    return dataset[list(np.arange(mask.shape[0])[mask])]
