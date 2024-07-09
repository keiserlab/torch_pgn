"""Hyperparameter optimization. Adapted from:
https://github.com/chemprop/chemprop/blob/master/chemprop/hyperparameter_optimization.py"""

from hyperopt import fmin, hp, tpe

import numpy as np

from copy import deepcopy
import os.path as osp
import os
import json
import datetime

from torch_pgn.train.Trainer import Trainer
from torch_pgn.args import HyperoptArgs


INT_KEYS = ['ffn_hidden_size', 'fp_dim', 'ffn_num_layers', 'depth', 'num_blocks', 'int_emb_size', 'hidden_channels',
            'basis_emb_size', 'out_emb_channels', 'num_spherical', 'num_radial', 'cutoff', 'envelope_exponent']

def hyperopt(args):
    """
    Runs hyperparmeter optimization.
    :param args: The arguments class containing the arguments used for optimization
    and training.
    :return: None
    """
    SPACES = {
        'ffn_hidden_size': hp.quniform('ffn_hidden_size', low=200, high=1400, q=200),
        'depth': hp.quniform('depth', low=3, high=6, q=1),
        'dropout_prob': hp.quniform('dropout_prob', low=0.0, high=0.4, q=0.1),
        'ffn_num_layers': hp.quniform('ffn_num_layers', low=1, high=5, q=1),
        'fp_dim': hp.choice('fp_dim', [1024, 4096, 8192]),
        'lr': hp.loguniform('learning_rate', np.log(1e-6), np.log(1e-2)),
        'num_blocks': hp.quniform('num_blocks', low=4, high=8, q=1),
        'int_emb_size': hp.choice('int_emb_size', [32, 64]),
        'nn_conv_internal_dim': hp.choice('nn_conv_internal_dim', [32,64]),
        'basis_emb_size': hp.choice('basis_emb_size', [32, 64, 128]),
        'out_emb_channels': hp.choice('out_emb_channels', [32, 64, 128]),
        'num_spherical': hp.quniform('num_spherical', low=4, high=10, q=1),
        'num_radial': hp.quniform('num_radial', low=4, high=16, q=1),
        'cutoff': hp.quniform('cutoff', low=4.5, high=10, q=0.5),
        'envelope_exponent': hp.quniform('envelop_exponent', low=3, high=6, q=1)
    }


    results = []
    trainer = Trainer(args)
    trainer.load_data()

    SPACE = {}
    for key in args.search_keys:
        SPACE[key] = SPACES[key]

    def objective(hyperparams):

        for key in INT_KEYS:
            if key in hyperparams:
                hyperparams[key] = int(hyperparams[key])

        hyper_args = deepcopy(args)

        current_time = datetime.datetime.now()
        folder_name = '_'.join(f'{key}_{value}' for key, value in hyperparams.items()).replace('.', 'p') + '_'+ str(current_time.microsecond)
        hyper_args.save_dir = osp.join(hyper_args.save_dir, folder_name)
        os.mkdir(hyper_args.save_dir)


        for key, value in hyperparams.items():
            setattr(hyper_args, key, value)

        # Set hyperparameter optimization args without reloading working_data
        trainer.set_hyperopt_args(hyper_args)
        # Run training using hyper_args
        trainer.run_training()
        # Retrieve the validation score from this round of training
        score = trainer.get_score()

        results.append({
            'score': score,
            'hyperparams': hyperparams
        })

        return (1 if hyper_args.minimize_score else -1) * score

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=args.num_iters, rstate=np.random.default_rng(args.seed))

    results = [result for result in results if not np.isnan(result['score'])]
    best_result = min(results, key=lambda result: (1 if args.minimize_score else -1) * result['score'])

    result_path = osp.join(args.save_dir, 'hyperopt_result.json')

    with open(result_path, 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)


def hyperparameter_optimization():
    """Processes hyperparameter optimization arguements and initiates an optimization run using the specified parameters.

    This function serves as an entry point for the command torch_pgn_hyperparameter_optimization in the command line
    """
    args = HyperoptArgs()
    args.parse_args()
    args.process_args()
    hyperopt(args)