import sys
sys.path.insert(0, "/srv/home/zgaleday/pgn")

from pgn.args import TrainArgs, HyperoptArgs, DataArgs
from pgn.load_data import process_raw
from scripts.generate_final_correlations import generate_final_correlations

import os
import os.path as osp

def generate_datasets(raw_path_1, raw_path_2, data_path, dataset_type,
                      radii=None,
                      lig_depth=None, receptor_depth=(1, 2, 3, 4, 5, 6)):
    if radii is None:
        radii = [4.5]
    if lig_depth is None:
        lig_depth = [-1]
    if receptor_depth is None:
        receptor_depth = [4]

    for radius in radii:
        for ld in lig_depth:
            for rd in receptor_depth:
                current_dir = osp.join(data_path, 'radius{0}_lig_depth_{1}_receptor_depth_{2}'.
                                       format(str(radius), str(ld), str(rd)))
                os.mkdir(current_dir)
                args = DataArgs()
                if dataset_type == 'many_v_many':
                    args.from_dict({'raw_data_path': raw_path_1,
                                    'label_file': raw_path_2,
                                    'data_path': current_dir,
                                    'dataset_type': dataset_type,
                                    'split_type': 'random',
                                    'save_plots': True,
                                    'proximity_radius': radius,
                                    'ligand_depth': ld,
                                    'receptor_depth': rd,
                                    'num_workers': 64
                                    })
                else:
                    args.from_dict({'raw_mol_path': raw_path_1,
                                    'raw_pdb_path': raw_path_2,
                                    'data_path': current_dir,
                                    'dataset_type': dataset_type,
                                    'split_type': 'random',
                                    'save_plots': False,
                                    'proximity_radius': rd,
                                    'ligand_depth': ld,
                                    'receptor_depth': receptor_depth,
                                    'num_workers': 64
                                    })

                args.process_args()

                process_raw(args)


def generate_repeats(data_path, checkpoint_path, save_dir, split_dir, device, epoch=250):
    for dir in os.listdir(data_path):
        try:
            current_dir = osp.join(save_dir, dir)
            os.mkdir(current_dir)
            data_dir = osp.join(data_path, dir)
            generate_final_correlations(checkpoint_path, current_dir, split_dir, device, data_path=data_dir, repeats=3, epochs=300)
        except:
            pass

if __name__ == '__main__':
    raw_data_path = sys.argv[1]
    raw_label_file = sys.argv[2]
    data_path = sys.argv[3]
    dataset_type = 'many_v_many'
    if len(sys.argv) > 4:
        dataset_type = sys.argv[4]
    try:
        generate_datasets(raw_data_path, raw_label_file, data_path, dataset_type)
    except:
        pass
    if len(sys.argv) > 5:
        checkpoint_dir = sys.argv[5]
        model_path = sys.argv[6]
        split_dir = sys.argv[7]
        device = sys.argv[8]
        generate_repeats(data_path, checkpoint_dir, model_path, split_dir, device)
