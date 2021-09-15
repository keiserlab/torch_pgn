import pandas as pd
import numpy as np
import oddt.pandas as opd

TEST_FILE = '/srv/home/zgaleday/IG_data/raw_data/d4_test_compounds/chembl_binders_chunk_map.csv'
MOL_FILE = '/srv/home/zgaleday/IG_data/raw_data/d4_test_compounds/chembl_experimental.mol2'
OUTFILE = '/srv/home/zgaleday/IG_data/raw_data/d4_test_compounds/chembl_experimental_bestpos.mol2'

test_df = pd.read_csv(TEST_FILE)
mol2_df = opd.read_mol2(MOL_FILE, skip_bad_mols=True)
mol2_df['best_energy'] = mol2_df.groupby(['mol_name'])['Total Energy'].transform(min)
mol2_df = mol2_df[mol2_df['Total Energy'] == mol2_df['best_energy']]
mol2_df.drop(columns='best_energy')
mol2_df.to_mol2(OUTFILE)
