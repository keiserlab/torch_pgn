import sys
sys.path.insert(0, "/srv/home/zgaleday/pgn")

from pgn.train.Trainer import Trainer
from pgn.train.train_utils import load_checkpoint
from pgn.data.load_data import _load_splits, _split_data
from pgn.data.data_utils import normalize_targets, normalize_distance

import numpy as np
import pandas as pd
import os
import os.path as osp

DATASET_SIZES = [1000, 5000, 10000, 25000, 50000]


def test_subsets(source_path, split_path, output_dir, device, subset_size=DATASET_SIZES, repeats=5):
    """
    A method to test the effect of dataset size on the performance of a model. Takes the result of generate_final_correlations
    as the input and generates a subset with the same test set of the training data. Repeats reruns of this are done.
    :param source_path:
    :param split_path:
    :param subset_size:
    :return: None
    """
    val_evals = []
    label_stats = []

    checkpoint_path = osp.join(source_path, 'repeat_0', 'best_checkpoint.pt')
    args = load_checkpoint(checkpoint_path, device=device, return_args=True)[1]
    args.construct_graphs = False
    args.split_type = 'defined_test'
    args.split_dir = split_path
    args.mode = 'evaluate'
    args.data_path = '~/IG_data/d4_graphs_pgn'
    args.load_test = True
    args.num_workers = 0
    args.cross_validate = False
    trainer = Trainer(args)
    trainer.load_data()
    for subset_size in DATASET_SIZES:
        subset_dir = osp.join(output_dir, 'subset_{0}'.format(subset_size))
        os.mkdir(subset_dir)
        for repeat in range(repeats):

            train_names, valid_names, test_names = _load_splits(split_path)
            train_names = np.hstack((train_names, valid_names))

            args.seed = np.random.randint(0, 1e4)
            rand = np.random.RandomState(args.seed)
            permutations = rand.permutation(len(train_names))

            train_index = permutations[:subset_size]

            if args.normalize_targets:
                trainer.train_data.data.y, label_stats = normalize_targets(trainer.train_data, index=train_index)
                args.label_mean, args.label_std = label_stats

            if args.normalize_dist:
                trainer.train_data.data.edge_attr, dist_stats = normalize_distance(trainer.train_data, args=args,
                                                                              index=train_index)
                args.distance_mean, args.distance_std = dist_stats

            repeat_dir = osp.join(subset_dir, 'repeat_{0}'.format(repeat))
            os.mkdir(repeat_dir)
            args.save_dir = repeat_dir

            trainer.args = args
            trainer.run_training()

            val_evals.append(trainer.valid_eval)
            label_stats.append((float(args.label_mean), float(args.label_std)))

            if args.normalize_targets:
                trainer.train_data.data.y = (trainer.train_data.data.y * args.label_std) + args.label_mean

            if args.normalize_dist:
                trainer.train_data.data.edge_attr[:, 0] = (trainer.train_data.data.edge_attr[:,
                                                   0] * args.distance_std) + args.distance_mean
        df = _format_evals(val_evals, label_stats)
        df.to_csv(osp.join(subset_dir, 'eval_stats.csv'))


def _format_evals(val_evals, label_stats):
    evals = {}
    for key in val_evals[0].keys():
        evals['test_' + key] = []
    evals['mean'] = []
    evals['std'] = []
    for idx in range(len(val_evals)):
        evals['mean'].append(label_stats[idx][0])
        evals['std'].append(label_stats[idx][1])
        for key in val_evals[idx].keys():
            evals['test_' + key].append(val_evals[idx][key])
    return pd.DataFrame(evals)

if __name__ == '__main__':
    source_path = sys.argv[1]
    final_path = sys.argv[2]
    split_path = sys.argv[3]
    device = sys.argv[4]
    test_subsets(source_path, split_path, final_path, device)