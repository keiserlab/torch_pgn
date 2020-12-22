from pgn.train.run_training import run_training
from pgn.train.hyperopt import hyperopt
from pgn.args import TrainArgs, HyperoptArgs

def test_train():
    args = TrainArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/tests/data/toy_out',
                    'dataset_type': 'many_v_many',
                    'split_type': 'random',
                    'construct_graphs': False,
                    'save_dir': '/Users/student/git/pgn/tests/output',
                    'epochs': 2,
                    'validation_percent': 0.2,
                    'test_percent': 0.2,
                    'save_splits': True,
                    'split_dir': '/Users/student/git/pgn/tests/splits',
                    'memory_stats': True,
                    'cv_folds': 2
                    })
    args.process_args()

    run_training(args)


def test_hyperopt():
    args = HyperoptArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/tests/data/toy_out',
                    'dataset_type': 'many_v_many',
                    'split_type': 'random',
                    'construct_graphs': False,
                    'save_dir': '/Users/student/git/pgn/tests/hyperopt_output',
                    'epochs': 2,
                    'save_splits': True,
                    'cv_folds': 5,
                    'num_iters': 2
                    })
    args.process_args()

    hyperopt(args)

test_train()