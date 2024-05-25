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
DATA_PATH = sys.argv[2]
SAVE_DIR = sys.argv[3]
STUDY_NAME = sys.argv[4]
NUM_TRIALS = int(sys.argv[5])
SPLIT_DIR = sys.argv[6]

def objective(trial):
    space = {}
    if ENCODER == 'dimenet++':
        space['lr'] = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
        space['num_blocks'] = trial.suggest_int("num_blocks", 2, 6)  # default 5
        space['int_emb_size'] = trial.suggest_categorical("int_emb_size", [32, 64])  # iteraction block embedding, default 64
        space['nn_conv_internal_dim'] = trial.suggest_categorical("nn_conv_internal_dim", [32, 64])  # hidden embedding size, default 64
        space['basis_emb_size'] = trial.suggest_categorical("basis_emb_size", [32, 64, 128])  # 64
        space['out_emb_channels'] = trial.suggest_categorical("out_emb_channels", [32, 64, 128])  # size of embedding in output block, default 64
        space['num_spherical'] = trial.suggest_int("num_spherical", 4, 10)  # 6
        space['num_radial'] = trial.suggest_int("num_radial", 4, 10)  # 6

    args = HyperoptArgs()

    if 'd4' in DATA_PATH:
        dataset_type = 'one_v_many'
    else:
        dataset_type = 'many_v_many'

    args.from_dict({'data_path': DATA_PATH,
                    'search_keys': list(space.keys()),
                    'encoder_type': ENCODER,
                    'dataset_type': dataset_type,
                    'construct_graphs': False,
                    'fp_dim': 1024 * 16,
                    'split_type': 'defined_test',
                    'split_dir': SPLIT_DIR,
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
    study = optuna.load_study(study_name=STUDY_NAME, storage=f'sqlite:///{SAVE_DIR}/{STUDY_NAME}.db')
    study.optimize(objective, n_trials=NUM_TRIALS)