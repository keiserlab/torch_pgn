import os
import sys
import argparse

sys.path.insert(0, "/srv/home/zgaleday/pgn")
from pgn.train.run_training import run_training
from pgn.train.hyperopt import hyperopt
from pgn.args import TrainArgs, HyperoptArgs

DATASETS = ['pdbbind_general', 'pdbbind_refined', 'd4']
# SPLIT_TYPES = ['random', 'scaffold', 'protein']
ARCHITECTURES = ['pgn', 'ggnet', 'dmpnn', 'fp', 'dimenet++']
DATASET_PATHS = {'pdbbind_general': '/srv/home/zgaleday/IG_data/pdbbind_general_pgn',
                 'pdbbind_refined': '/srv/home/zgaleday/IG_data/pdb_bind_random',
                 'PLEC_pdbbind_general': '/srv/home/zgaleday/IG_data/pgn_pdbbind_general_plec_16384',
                 'PLEC_pdbbind_refined': '/srv/home/zgaleday/IG_data/pgn_pdbbind_refined_plec_16384',
                 'd4': '/srv/home/zgaleday/IG_data/d4_graphs_pgn',
                 'PLEC_d4': '/srv/home/zgaleday/IG_data/pgn_d4_plec_16384'}
SEARCH_KEYS = {
    'pgn': ['ffn_hidden_size', 'ffn_num_layers', 'dropout', 'fp_dim'],
    'ggnet': ['ffn_hidden_size', 'ffn_num_layers', 'dropout', 'depth'],
    'dmpnn': ['ffn_hidden_size', 'ffn_num_layers', 'dropout', 'depth'],
    'fp': ['ffn_hidden_size', 'ffn_num_layers', 'dropout'],
    'dimenet++': ['num_blocks', 'int_emb_size', 'nn_conv_internal_dim',
            'basis_emb_size', 'out_emb_channels', 'num_radial', 'num_spherical', 'lr']
}


def hyperopt_wrapper(save_dir, dataset, architecture, epochs, device, batch_size=128, multi_gpu=False, cv_folds=3, num_iters=25):

    data_path = DATASET_PATHS[dataset]
    search_keys = SEARCH_KEYS[architecture]
    encoder_type = architecture

    args = HyperoptArgs()

    args.from_dict({'data_path': data_path,
                    'search_keys': search_keys,
                    'encoder_type': encoder_type,
                    'fp_dim': 1024 * 16,
                    'split_type': 'random',
                    'save_dir': save_dir,
                    'device': device,
                    'epochs': epochs,
                    'cv_folds': cv_folds,
                    'save_splits': True,
                    'num_iters': num_iters,
                    'num_workers': 0,
                    'batch_size': batch_size,
                    'weight_decay': False})
    args.process_args()

    print(args)

    hyperopt(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Hyperopt script',
        description='Script used to perform hyper parameter optimization for PGN Paper')
    parser.add_argument('save_dir', help='path to save output of hyperparameter optimization. Must be empty.')
    parser.add_argument('dataset', help='Dataset type to be used in optimization', choices=DATASETS)
    parser.add_argument('architecture', help='Type of encoder to be used in optimization', choices=ARCHITECTURES)
    parser.add_argument('epochs')
    parser.add_argument('device')
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--cv_folds', default=3)
    parser.add_argument('--num_iters', default=25)
    args = parser.parse_args()
    hyperopt_wrapper(**vars(args))