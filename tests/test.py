from pgn.train.run_training import run_training
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
                    'save_splits': True
                    })
    args.process_args()

    run_training(args)


test_train()