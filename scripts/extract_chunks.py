import pandas as pd
import numpy as np
import gzip
import os.path as osp
import os
import shutil
import oddt.pandas as opd
from tqdm import tqdm

TEST_FILE = '/srv/home/zgaleday/IG_data/raw_data/d4_mmgbsa/d4_mmgbsa_set_chunk_map.csv'
SCREEN_DIRECTORY = '/srv/nas/mk2/projects/D4_screen/'
TEST_CHUNK = 'vs_run1_chunk28150'
OUTDIR = '/srv/home/zgaleday/IG_data/raw_data/d4_mmgbsa'

def parse_chunk(chunk_id, mol_names):
    temp_file = osp.join(OUTDIR, 'temp.mol2')
    with gzip.open(osp.join(SCREEN_DIRECTORY, chunk_id, 'test.mol2.gz'), 'rb') as f:
        with open(temp_file, 'wb') as f_out:
            shutil.copyfileobj(f, f_out)
    chunk = opd.read_mol2(temp_file)
    chunk['mol_name'] = chunk['mol_name'].apply(lambda x: x.split()[0])
    os.remove(temp_file)
    return chunk[chunk['mol_name'].isin(mol_names)]

exp_df = pd.read_csv(TEST_FILE)
grouped_df = exp_df.groupby('chunk')['name'].apply(list)

mol2_list = []
failed_chunks = []
for chunk in tqdm(grouped_df.index):
    try:
        mol2_list.append(parse_chunk(chunk, grouped_df[chunk]))
    except:
        failed_chunks.append(chunk)


#full_mol2 = mol2_list[0]
full_mol2 = pd.concat(mol2_list)
print(full_mol2)

full_mol2.to_mol2(osp.join(OUTDIR, 'd4_mmgbsa_set.mol2'))
np.save(osp.join(OUTDIR, 'd4_mmgbsa_failed.npy'), np.array(failed_chunks))
print(failed_chunks)
