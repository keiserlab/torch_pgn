from torch_pgn.data.data_utils import (parse_transforms,
                                       normalize_targets, normalize_distance
                                       )
from torch_pgn.data.ProximityGraphDataset import ProximityGraphDataset
from torch_pgn.data.FingerprintDataset import FingerprintDataset

import torch
import numpy as np
import os.path as osp
import os

# TODO: Add a little bit more though into how we want to do defined splits. Probably have an option for only having
#  a defined test set i.e. defined_test option in split type

# TODO: Change name to reflect loading FPs
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
    norm_dist = args.normalize_dist and args.include_dist
    load_test = args.load_test
    cross_validate = args.cross_validate
    dataset_type = args.dataset_type

    if cross_validate:
        return _load_data_cross_validation(args)

    torch.manual_seed(seed)

    if split_type == 'random':
        if dataset_type == 'fp':
            train_dataset = FingerprintDataset(args)
        else:
            train_dataset = ProximityGraphDataset(args)
            train_dataset._data = transforms(train_dataset._data)

        num_examples = len(train_dataset._data.name)

        rand = np.random.RandomState(args.seed)
        permutations = rand.permutation(num_examples)

        valid_begin, valid_end = 0, int(args.validation_percent * num_examples)
        test_begin, test_end = valid_end, int(args.test_percent * num_examples) + valid_end
        train_begin, train_end = test_end, num_examples

        if args.mode == 'experiment':
            train_index = permutations[train_begin:train_end]
            valid_index = permutations[valid_begin:valid_end]
        else:
            train_index = permutations[train_begin:valid_end]
        test_index = permutations[test_begin:test_end]

        if args.mode == 'experiment':
            validation_dataset = train_dataset[list(valid_index)]
        test_dataset = train_dataset[list(test_index)]
        train_dataset = train_dataset[list(train_index)]

        if args.save_splits:
            if args.mode == 'experiment':
                _save_splits(args.save_dir, (train_dataset, train_index),
                             validation=(validation_dataset, valid_index),
                             test=(test_dataset, test_index))
            else:
                _save_splits(args.save_dir, (train_dataset, train_index),
                             test=(test_dataset, test_index))

    elif split_type == 'defined_test':
        # TODO: make sure error handling of no split dir happens somewhere
        split_dir = args.split_dir
        train_names, valid_names, test_names = _load_splits(split_dir)
        train_names = np.hstack((train_names, valid_names))

        if valid_names is not None:
            num_examples = train_names.shape[0] + valid_names.shape[0] + test_names.shape[0]
        else:
            num_examples = train_names.shape[0] + test_names.shape[0]

        if dataset_type == 'fp':
            train_dataset = FingerprintDataset(args)
        else:
            train_dataset = ProximityGraphDataset(args)
            train_dataset.data = transforms(train_dataset.data)

        test_index = _split_data(train_dataset, test_names)
        train_index = _split_data(train_dataset, train_names)



        if args.mode == 'experiment':
            valid_end = int(args.validation_percent * num_examples)
            train_begin = valid_end
            rand = np.random.RandomState(args.seed)
            permutations = rand.permutation(len(train_index))
            valid_index = list(np.array(train_index)[list(permutations[:valid_end])])
            train_index = list(np.array(train_index)[list(permutations[train_begin:])])
            validation_dataset = train_dataset[valid_index]


        test_dataset = train_dataset[test_index]
        train_dataset = train_dataset[train_index]

    elif split_type == 'defined':

        split_dir = args.split_dir
        train_names, valid_names, test_names = _load_splits(split_dir)

        if dataset_type == 'fp':
            train_dataset = FingerprintDataset(args)
        else:
            train_dataset = ProximityGraphDataset(args)
            train_dataset.data = transforms(train_dataset.data)

        test_index = _split_data(train_dataset, test_names)
        train_index = _split_data(train_dataset, train_names)
        valid_index = _split_data(train_dataset, valid_names)

        if args.mode == 'experiment':
            validation_dataset = train_dataset[valid_index]
            train_dataset = train_dataset[train_index]
        else:
            train_dataset = [train_index + valid_index]

        test_dataset = train_dataset[test_index]




    else:
        raise ValueError('Invalid dataset type. Please choose from <random> or <defined>')

    if dataset_type != 'fp':
        args.node_dim = train_dataset._data.x.numpy().shape[1]
        args.edge_dim = train_dataset._data.edge_attr.numpy().shape[1]

    if norm_targets:
        train_dataset._data.y, label_stats = normalize_targets(train_dataset, index=train_index)
        args.label_mean, args.label_std = label_stats


    if norm_dist:
        train_dataset._data.edge_attr, dist_stats = normalize_distance(train_dataset, args=args, index=train_index)
        args.distance_mean, args.distance_std = dist_stats


    if args.mode == 'experiment':
        return train_dataset, validation_dataset
    else:
        return train_dataset, test_dataset


def _load_data_cross_validation(args):
    """
    Helper function to keep main function from getting to complicated and clutter. Handles the loading and returning
    or working_data when cross-validation will be used to train and evaluate/select the model.
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
    norm_dist = args.normalize_dist and args.include_dist
    load_test = args.load_test
    dataset_type = args.dataset_type

    torch.manual_seed(seed)

    if split_type == 'random':

        if dataset_type == 'fp':
            train_dataset = FingerprintDataset(args)
        else:
            train_dataset = ProximityGraphDataset(args)
            train_dataset.data = transforms(train_dataset.data)

        num_examples = len(train_dataset.data.name)

        rand = np.random.RandomState(args.seed)
        permutations = rand.permutation(num_examples)

        test_begin, test_end = 0, int(args.test_percent * num_examples)
        train_begin, train_end = test_end, num_examples

        train_index = list(permutations[train_begin:train_end])
        test_index = list(permutations[test_begin:test_end])

        if args.save_splits:
            _save_splits(args.save_dir, (train_dataset, train_index),
                         test=(train_dataset[test_index], test_index))

    elif split_type == 'defined_test':
        # TODO: make sure error handling of no split dir happens somewhere
        split_dir = args.split_dir
        train_names, valid_names, test_names = _load_splits(split_dir)
        train_names = np.hstack((train_names, valid_names))

        if valid_names is not None:
            num_examples = train_names.shape[0] + valid_names.shape[0] + test_names.shape[0]
        else:
            num_examples = train_names.shape[0] + test_names.shape[0]

        if dataset_type == 'fp':
            train_dataset = FingerprintDataset(args)
        else:
            train_dataset = ProximityGraphDataset(args)
            train_dataset.data = transforms(train_dataset.data)

        test_index = _split_data(train_dataset, test_names)
        train_index = _split_data(train_dataset, train_names)

        if args.save_splits:
            _save_splits(args.save_dir, (train_dataset, train_index),
                         test=(train_dataset[test_index], test_index))


    elif split_type == 'defined':

        train_begin, train_end = args.train_splits

        if dataset_type == 'fp':
            train_dataset = FingerprintDataset(args)
        else:
            train_dataset = ProximityGraphDataset(args)
            train_dataset.data = transforms(train_dataset.data)

        train_dataset = train_dataset[train_begin:train_end]

        if load_test:
            test_dataset = ProximityGraphDataset(data_path, include_dist=include_dist, mode='test')
            test_dataset.data = transforms(test_dataset.data)

    else:
        raise ValueError('Invalid dataset type. Please choose from <random> or <defined>')

    if dataset_type != 'fp':
        args.node_dim = train_dataset.data.x.numpy().shape[1]
        args.edge_dim = train_dataset.data.edge_attr.numpy().shape[1]

    args.train_index = train_index

    return train_dataset


def _save_splits(base_dir, train, test, validation=None):
    """
    Helper function to save the working_data splits.
    :param base_dir: The base_directory to write the splits directory containing the saved splits
    :param train: Tuple(train_dataset, train_index)
    :param validation: Tuple(validation_dataset, validation_index) .
    :param test: Tuple(test_dataset, test_index).
    """
    split_dir = osp.join(base_dir, 'splits')
    os.mkdir(split_dir)
    train_data, train_index = train
    np.save(osp.join(split_dir, 'train.npy'), np.array(train_data._data.name)[train_index])
    if validation is not None:
        val_data, val_index = validation
        np.save(osp.join(split_dir, 'validation.npy'), np.array(val_data._data.name)[val_index])
    if test is not None:
        test_data, test_index = test
        np.save(osp.join(split_dir, 'test.npy'), np.array(test_data._data.name)[test_index])


def _load_splits(split_dir):
    """
    Loads the splits from split_dir
    :param split_dir: Directory containing train.npy, validation.npy and test.npy
    :return: the name lists for each of the defined splits (train, valid, test).
    """
    train_path = osp.join(split_dir, 'train.npy')
    try:
        valid_path = osp.join(split_dir, 'validation.npy')
        valid_name = np.load(valid_path)
    except:
        valid_name = None
        print("No validation_splits")
    test_path = osp.join(split_dir, 'test.npy')

    train_name = np.load(train_path)
    test_name = np.load(test_path)

    return train_name, valid_name, test_name

#TODO: Split this into different file.
def _split_data(dataset, names):
    mask = np.vectorize(lambda value: value in names)(np.array(dataset.data.name))
    index = list(np.arange(mask.shape[0])[mask])
    return index
