import pandas as pd
import numpy as np
import oddt.pandas as opd

TEST_FILE = '/srv/home/zgaleday/IG_data/raw_data/d4_900k_diverse/900k_diverse_chunk_map_-55_to_-49.csv'
MOL_FILE = '/srv/home/zgaleday/IG_data/raw_data/d4_900k_diverse/diverse_900k_ds_-55to-49.mol2'
OUTFILE = '/srv/home/zgaleday/IG_data/raw_data/d4_test_compounds/diverse_900k_ds_-55to-49_bestpos.mol2'

test_df = pd.read_csv(TEST_FILE)
mol2_df = opd.read_mol2(MOL_FILE, skip_bad_mols=True)
mol2_df['best_energy'] = mol2_df.groupby(['mol_name'])['Total Energy'].transform(min)
mol2_df = mol2_df[mol2_df['Total Energy'] == mol2_df['best_energy']]
mol2_df.drop(columns='best_energy')
mol2_df.to_mol2(OUTFILE)
