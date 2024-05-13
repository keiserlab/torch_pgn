sys.path.insert(0, "/srv/home/zgaleday/pgn")

from pgn.args import TrainArgs, HyperoptArgs, DataArgs
from pgn.load_data import process_raw
import sys
import os
import os.path as osp

def generate_datasets(raw_path, raw_label_file, data_path, dataset_type,
                      radii=(2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6),
                      lig_depth=-1, receptor_depth=4):

    for radius in radii:
        current_dir = osp.join(data_path, 'radius{0}_lig_depth_{1}_receptor_depth_{2}'.
                               format(str(radius), str(lig_depth), str(receptor_depth)))
        os.mkdir(current_dir)
        args = DataArgs()
        args.from_dict({'raw_data_path': raw_path,
                        'label_file': raw_label_file,
                        'data_path': current_dir,
                        'dataset_type': dataset_type,
                        'split_type': 'random',
                        'save_plots': True,
                        'proximity_radius': radius,
                        'ligand_depth': lig_depth,
                        'receptor_depth': receptor_depth
                        })

        args.process_args()

        process_raw(args)

if __name__ == '__main__':
    raw_data_path = sys.argv[1]
    raw_label_file = sys.argv[2]
    data_path = sys.argv[3]
    dataset_type = 'many_v_many'
    if len(sys.argv) > 4:
        dataset_type = sys.argv[4]
    generate_datasets(raw_data_path, data_path, dataset_type)