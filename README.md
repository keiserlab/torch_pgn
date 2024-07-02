# torch_pgn
Proximity Graph Networks (torch_pgn) is a pytorch toolkit allowing for the modular application of multiple different encoder architectures to cheminformatic tasks centered around protein-ligand complexes. Alpha version of documentation is available at: <url>https://torch-pgn.readthedocs.io/en/latest/index.html<url>.

## Installation
torch-pgn either be installed from PyPi using the pip command or from source. We assume that all users are using conda, if you do not have conda, please install Miniconda from <url>https://conda.io/miniconda.html<url>.

### Installation using pip (cpu only)
1. `conda create --name torch_pgn python=3.7`
2. `conda activate torch_pgn`
3. `pip install torch_pgn`
4. `conda install pytorch-sparse -c pyg`
5. `conda install -c conda-forge openbabel`

### Installation using pip (cuda)
1. `conda create --name torch_pgn python=3.7`
2. `conda activate torch_pgn`
3. `conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`
4. `conda install pyg -c pyg`
5. `conda install pytorch-sparse -c pyg`
6. `conda install -c conda-forge openbabel`
7. `pip install torch_pgn`

### Installation from source
1. `git clone https://github.com/keiserlab/torch_pgn/torch_pgn.git`
2. `cd torch_pgn`
3. `conda env create -f environment.yml`
4. `conda activate torch_pgn`
5. `pip install -e`
