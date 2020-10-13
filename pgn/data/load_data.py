from pgn.data.data_utils import (parse_transforms,
                                 normalize_targets, normalize_distance
                                 )
from pgn.data.ProximityGraphDataset import ProximityGraphDataset

import torch


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
    dataset_type = args.dataset_type
    norm_targets = args.normalize_targets
    include_dist = args.include_dist
    norm_dist = args.normalize_dist
    load_test = args.load_test

    torch.manual_seed(seed)

    if dataset_type == 'combined':
        valid_begin, valid_end = args.validation_splits
        test_begin, test_end = args.test_splits
        train_begin, train_end = args.train_splits

        train_dataset = ProximityGraphDataset(data_path, include_dist=include_dist)
        train_dataset.data = transforms(train_dataset.data)

        #TODO: Fix edgecase the remove last example if last index is end of dataset
        validation_dataset = train_dataset[valid_begin:valid_end]
        test_dataset = train_dataset[test_begin: test_end]
        train_dataset = train_dataset[train_begin:train_end]

        if norm_targets:
            train_dataset.data.y, target_stats = normalize_targets(train_dataset, yield_stats=True)

        if norm_dist:
            train_dataset.data.edge_attr, dist_stats = normalize_distance(train_dataset, yield_stats=True)


    elif dataset_type == 'split':
        valid_begin, valid_end = args.validation_splits
        train_begin, train_end = args.train_splits

        train_dataset = ProximityGraphDataset(data_path, include_dist=include_dist)
        train_dataset.data = transforms(train_dataset.data)

        validation_dataset = train_dataset[valid_begin:valid_end]
        train_dataset = train_dataset[train_begin:train_end]

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
        raise ValueError('Invalid dataset type. Please choose from <combined> or <split>')

    #TODO: Need to write out splits and the statistics somehow. Probably best to use some sort of logger object??

    if load_test:
        return train_dataset, validation_dataset, test_dataset

    else:
        return train_dataset, validation_dataset