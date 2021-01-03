"""Hyperparameter optimization. Adapted from:
https://github.com/chemprop/chemprop/blob/master/chemprop/hyperparameter_optimization.py"""

from hyperopt import fmin, hp, tpe

import numpy as np

from copy import deepcopy
import os.path as osp
import os
import json

from pgn.train.Trainer import Trainer

SPACE = {
    'ffn_hidden_size': hp.quniform('ffn_hidden_size', low=200, high=2400, q=100),
    #'depth': hp.quniform('depth', low=2, high=6, q=1),
    'dropout': hp.quniform('dropout', low=0.0, high=0.4, q=0.05),
    'ffn_num_layers': hp.quniform('ffn_num_layers', low=1, high=5, q=1),
    'fp_dim': hp.quniform('fp_dim', low=1024, high=8192, q=1024)
}

INT_KEYS = ['ffn_hidden_size', 'fp_dim', 'ffn_num_layers']

def hyperopt(args):
    """
    Runs hyperparmeter optimization.
    :param args: The arguments class containing the arguments used for optimization
    and training.
    :return: None
    """
    results = []
    trainer = Trainer(args)
    trainer.load_data()

    def objective(hyperparams):

        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])

        hyper_args = deepcopy(args)

        folder_name = '_'.join(f'{key}_{value}' for key, value in hyperparams.items()).replace('.', 'p')
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
        print(trainer.valid_eval)

        results.append({
            'score': score,
            'hyperparams': hyperparams
        })

        return (1 if hyper_args.minimize_score else -1) * score

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=args.num_iters, rstate=np.random.RandomState(args.seed))

    results = [result for result in results if not np.isnan(result['score'])]
    best_result = min(results, key=lambda result: (1 if args.minimize_score else -1) * result['score'])

    result_path = osp.join(args.save_dir, 'hyperopt_result.json')

    with open(result_path, 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)



