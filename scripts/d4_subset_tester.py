import sys
sys.path.insert(0, "/srv/home/zgaleday/torch_pgn")

from torch_pgn.train.Trainer import Trainer
from torch_pgn.data.ProximityGraphDataset import ProximityGraphDataset
from torch_pgn.train.train_utils import load_checkpoint
from torch_pgn.data.load_data import _load_splits, _split_data, _save_splits
from torch_pgn.data.data_utils import normalize_targets, normalize_distance, parse_transforms

import numpy as np
import pandas as pd
import os
import os.path as osp

DATASET_SIZES = [1000, 2000, 3000, 4000]


def test_subsets(source_path, split_path, output_dir, device, data_path=None, subset_size=DATASET_SIZES, repeats=5):
    """
    A method to test the effect of dataset size on the performance of a model. Takes the result of generate_final_correlations
    as the input and generates a subset with the same test set of the training data. Repeats reruns of this are done.
    :param source_path:
    :param split_path:
    :param subset_size:
    :return: None
    """
    val_evals = []
    label_stat_list = []

    checkpoint_path = osp.join(source_path)
    args = load_checkpoint(checkpoint_path, device=device, return_args=True)[1]
    args.construct_graphs = False
    args.split_type = 'defined_test'
    args.split_dir = split_path
    args.mode = 'evaluate'
    if data_path is not None:
        args.data_path = data_path
    args.load_test = True
    args.num_workers = 0
    args.cross_validate = False
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
            train_names = train_names[list(train_index)]

            repeat_dir = osp.join(subset_dir, 'repeat_{0}'.format(repeat))
            os.mkdir(repeat_dir)
            args.save_dir = repeat_dir
            current_splits = osp.join(repeat_dir, 'splits')
            os.mkdir(current_splits)
            args.split_dir = current_splits
            np.save(osp.join(current_splits, 'train.npy'), np.array(train_names, dtype=str))
            np.save(osp.join(current_splits, 'test.npy'), np.array(test_names, dtype=str))

            trainer = Trainer(args)
            trainer.load_data()
            trainer.run_training()

            val_evals.append(trainer.valid_eval)
            label_stat_list.append((float(args.label_mean), float(args.label_std)))
            trainer = None
            del trainer


        df = _format_evals(val_evals, label_stat_list)
        df.to_csv(osp.join(subset_dir, 'eval_stats.csv'))
        val_evals = []
        label_stat_list = []


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
    if len(sys.argv) > 5:
        data_path = sys.argv[5]
    else:
        data_path = None
    test_subsets(source_path, split_path, final_path, device, data_path)
