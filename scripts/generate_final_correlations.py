from pgn.train.Trainer import Trainer
from pgn.train.train_utils import load_checkpoint
from pgn.train.run_training import run_training

import numpy as np

import os.path as osp
import os


def generate_final_correlations(checkpoint_path, final_path, split_path, device, repeats=5):
    """
    Loads the checkpoint. Gives a random seed and generates <repeats> models with the same hyperparamters and different initialization.
    :param checkpoint_path: The path of the checkpoint file to load the args from.
    :param final_path: The path to save the results into.
    :param split_path: The path where the splits used to train the model were saved. Only the testing split is used (to ensure no data contamination).
    :param device: device to use for training
    :param repeats: The number of trails to run.
    :return: None
    """
    val_evals = []
    test_evals = []
    label_stats = []
    args = load_checkpoint(checkpoint_path, device=device, return_args=True)[1]
    args.split_type = 'defined_test'
    args.split_dir = split_path
    args.load_test = True
    base_dir = final_path
    for iter in range(repeats):
        save_dir = osp.join(base_dir, 'repeat_{0}'.format(iter))
        os.mkdir(save_dir)
        args.save_dir = save_dir
        args.seed = np.random.randint(0, 1e4)
        args.epochs = 300
        trainer = run_training(args)
        val_evals.append(trainer.valid_eval)
        test_evals.append(trainer.test_eval)
        label_stats.append((args.label_mean, args.label_std))




