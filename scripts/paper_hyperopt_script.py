import os
import sys

sys.path.insert(0, "/srv/home/zgaleday/pgn")
from pgn.train.run_training import run_training
from pgn.train.hyperopt import hyperopt
from pgn.args import TrainArgs, HyperoptArgs

DATASETS = ['pdbbind_general', 'pdbbind_refined', 'd4']
# SPLIT_TYPES = ['random', 'scaffold', 'protein']
ARCHITECTURES = ['pgn', 'ggnet', 'dmpnn', 'fp']
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
    'fp': ['ffn_hidden_size', 'ffn_num_layers', 'dropout']
}


def hyperopt_wrapper(save_dir, dataset, architecture, epochs, batch_size=128, multi_gpu=False, device=None):

    data_path = DATASET_PATHS[dataset]
    search_keys = SEARCH_KEYS[dataset]
    encoder_type = architecture

    args = HyperoptArgs()

    args.from_dict({'data_path': '/srv/home/zgaleday/IG_data/pgn_pdbbind_refined_plec_16384/',
                    'search_keys': ['ffn_hidden_size', 'ffn_num_layers', 'dropout'],
                    'encoder_type': encoder_type,
                    'fp_dim': 1024 * 16,
                    'split_type': 'random',
                    'save_dir': save_dir,
                    'device': device,
                    'epochs': epochs,
                    'cv_folds': 5,
                    'save_splits': True,
                    'num_iters': len(search_keys) * 5,
                    'num_workers': 0,
                    'batch_size': 128,
                    'weight_decay': True})
    args.process_args()

    print(args)

    hyperopt(args)


if __name__ == '__main__':
    checkpoint_path = sys.argv[1]
    final_path = sys.argv[2]
    split_path = sys.argv[3]
    device = sys.argv[4]
    epochs = int(sys.argv[5])
    generate_final_correlations(checkpoint_path, final_path, split_path, device, epochs=epochs)