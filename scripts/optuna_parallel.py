import optuna
import sys
import os
import datetime
import os.path as osp

sys.path.insert(0, "/srv/home/zgaleday/pgn")
from pgn.args import HyperoptArgs
from pgn.train.run_training import run_training

DEVICE = 'cuda:' + sys.argv[1]
ENCODER = 'dimenet++'
DATA_PATH = '/srv/home/zgaleday/IG_data/d4_graphs_pgn'
SAVE_DIR = '/srv/home/zgaleday/models/pgn/review_experiments/dimenet_d4_optuna_final'

def objective(trial):
    space = {}
    if ENCODER == 'dimenet++':
        space['lr'] = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
        space['num_blocks'] = trial.suggest_int("num_blocks", 4, 8)  # default 5
        space['int_emb_size'] = trial.suggest_categorical("int_emb_size", [32, 64])  # iteraction block embedding, default 64
        space['nn_conv_internal_dim'] = trial.suggest_categorical("nn_conv_internal_dim", [32, 64])  # hidden embedding size, default 64
        space['basis_emb_size'] = trial.suggest_categorical("basis_emb_size", [32, 64, 128])  # 64
        space['out_emb_channels'] = trial.suggest_categorical("out_emb_channels", [32, 64, 128])  # size of embedding in output block, default 64
        space['num_spherical'] = trial.suggest_int("num_spherical", 4, 10)  # 6
        space['num_radial'] = trial.suggest_int("num_radial", 8, 16)  # 6

    args = HyperoptArgs()

    args.from_dict({'data_path': DATA_PATH,
                    'search_keys': list(space.keys()),
                    'encoder_type': ENCODER,
                    'dataset_type': 'one_v_many',
                    'construct_graphs': False,
                    'fp_dim': 1024 * 16,
                    'split_type': 'random',
                    'device': DEVICE,
                    'epochs': 300,
                    'cv_folds': 3,
                    'save_dir': SAVE_DIR,
                    'save_splits': True,
                    'num_iters': 25,
                    'num_workers': 0,
                    'batch_size': 128,
                    'weight_decay': False})
    args.process_args()

    current_time = datetime.datetime.now()
    folder_name = '_'.join(f'{key}_{value}' for key, value in space.items()).replace('.', 'p') + '_' + str(
    current_time.microsecond)
    args.save_dir = osp.join(args.save_dir, folder_name)
    os.mkdir(args.save_dir)

    for key, value in space.items():
        setattr(args, key, value)

    trainer = run_training(args)
    # Set hyperparameter optimization args without reloading working_data
    # Run training using hyper_args
    # Retrieve the validation score from this round of training
    score = trainer.get_score()
    return score


if __name__ == '__main__':
    study = optuna.load_study(study_name='d4_dimenet_final', storage=f'sqlite:///{SAVE_DIR}/d4_dimenet.db')
    study.optimize(objective, n_trials=25)