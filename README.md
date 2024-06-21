# torch_pgn
Proximity Graph Networks (torch_pgn) is a pytorch toolkit allowing for the modular application of multiple different encoder architectures to cheminformatic tasks centered around protein-ligand complexes.

## Installation
torch-pgn either be installed from PyPi using the pip command or from source. We assume that all users are using conda, if you do not have conda, please install Miniconda from <url>https://conda.io/miniconda.html<url>.

### Installation using pip
1. <code>conda create --name torch_pgn python=3.7</code>
2. <code>conda activate torch_pgn</code>
3. <code>pip install torch_pgn</code>
4. <code>conda install pytorch-sparse -c pyg</code>
5. <code>conda install -c conda-forge openbabel</code>

> [!NOTE]
> If you are using a gpu machine and run into issues with this installation method we suggest you remove pytorch and pyg and reinstall using conda as follows:
> 1. `conda remove pytorch`
> 2. `conda remove pyg`
> 3. `conda install pytorch`
> 4. `conda install pyg -c pyg`
> 5. `conda install pytorch-sparse -c pyg`

### Installation from source
1. `git clone https://github.com/keiserlab/torch_pgn/torch_pgn.git`
2. `cd torch_pgn`
3. `conda env create -f environment.yml`
4. `conda activate torch_pgn`
5. `pip install -e`
