from pgn.train.run_training import run_training
from pgn.train.hyperopt import hyperopt
from pgn.args import TrainArgs, HyperoptArgs, DataArgs
from pgn.load_data import process_raw
from scripts.generate_plec import generate_plec_pdbbind
from scripts.generate_final_correlations import generate_final_correlations

def test_pfp_train():
    args = TrainArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/tests/working_data/toy_out',
                    'dataset_type': 'many_v_many',
                    'split_type': 'random',
                    'construct_graphs': False,
                    'save_dir': '/Users/student/git/pgn/tests/output/pfp',
                    'epochs': 1,
                    'cv_folds': 3,
                    'validation_percent': 0.2,
                    'test_percent': 0.2,
                    'save_splits': True
                    })
    args.process_args()

    run_training(args)


def test_hyperopt():
    args = HyperoptArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/tests/working_data/toy_out',
                    'dataset_type': 'many_v_many',
                    'split_type': 'random',
                    'construct_graphs': False,
                    'save_dir': '/Users/student/git/pgn/tests/hyperopt_output',
                    'epochs': 2,
                    'save_splits': True,
                    'cv_folds': 2,
                    'num_iters': 2
                    })
    args.process_args()

    hyperopt(args)

def test_dmpnn_train():

    args = TrainArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/tests/working_data/toy_out',
                    'dataset_type': 'many_v_many',
                    'split_type': 'random',
                    'construct_graphs': False,
                    'save_dir': '/Users/student/git/pgn/tests/output',
                    'epochs': 5,
                    'validation_percent': 0.2,
                    'test_percent': 0.2,
                    'save_splits': True,
                    'split_dir': '/Users/student/git/pgn/tests/splits',
                    'encoder_type': 'dmpnn',
                    'cv_folds': 2
                    })
    args.process_args()

    run_training(args)


def test_ggnet_train():

    args = TrainArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/ManyVsManyToy',
                    'data_path': '/Users/student/git/pgn/tests/working_data/toy_out',
                    'dataset_type': 'many_v_many',
                    'split_type': 'random',
                    'construct_graphs': False,
                    'save_dir': '/Users/student/git/pgn/tests/output',
                    'epochs': 5,
                    'validation_percent': 0.2,
                    'test_percent': 0.2,
                    'save_splits': True,
                    'split_dir': '/Users/student/git/pgn/tests/splits',
                    'encoder_type': 'ggnet'
                    })
    args.process_args()

    run_training(args)

def test_fp_dataloading():

    args = DataArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/FPToy/formated_plec_toy.csv',
                    'data_path': '/Users/student/git/pgn/tests/working_data/fp_out',
                    'split_type': 'random',
                    'dataset_type': 'fp',
                    'fp_format': 'sparse',
                    'fp_dim': 1024 * 16,
                    'label_col': 2
                    })

    args.process_args()

    process_raw(args)


def test_fp_train():

    args = TrainArgs()

    args.from_dict({'raw_data_path': '/Users/student/git/pgn/tests/working_data/FPToy/formated_plec_toy.csv',
                    'data_path': '/Users/student/git/pgn/tests/working_data/fp_out',
                    'construct_graphs': False,
                    'split_type': 'random',
                    'dataset_type': 'fp',
                    'encoder_type': 'fp',
                    'fp_format': 'sparse',
                    'fp_dim': 1024 * 16,
                    'label_col': 2,
                    'save_dir': '/Users/student/git/pgn/tests/output',
                    'epochs': 5,
                    'validation_percent': 0.2,
                    'test_percent': 0.2,
                    'save_splits': True})

    args.process_args()

    run_training(args)

def test_generate_plec_pdbbind():

    generate_plec_pdbbind('working_data/ManyVsManyToy', 'working_data/ManyVsManyToy/index/INDEX_general_PL_data.2019')

def test_generate_final_correlations():
    generate_final_correlations('/Users/student/git/pgn/tests/output/pfp/cv_fold_0/best_checkpoint.pt',
                                '/Users/student/git/pgn/tests/output/final_corr',
                                split_path='/Users/student/git/pgn/tests/output/pfp/splits',
                                device='cpu',
                                epochs=5)

test_generate_final_correlations()