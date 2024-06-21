from setuptools import setup, find_packages

__version__ = "0.1.2"

# Load README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='torch_pgn',
    version=__version__,
    description='Proximity Graph Networks: Predicting ligand affinity with Message Passing Neural Networks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keiserlab/torch_pgn",
    author='Keiser Lab',
    author_email='keiser@keiserlab.org',
    license="MIT",
    packages=find_packages(),
    package_data={'torch_pgn': ["py.typed"]},
    install_requires=[
        "torch>=1.7.0",
        "torch_geometric>=1.6.3",
        "networkx>=2.5",
        "typed-argument-parser==1.6.1",
        "pandas==1.1.5",
        "rdkit>=2020.09.3",
        "numpy>=1.19.2,<=1.21.6",
        "tqdm>=4.54.1",
        "tensorboard>2.0",
        "scipy<1.7",
        "scikit-learn>=0.23.2",
        "hyperopt>=0.2.5",
        "sympy>=1.10.1",
        "optuna>=3.6.1",
        "matplotlib>=3.0.0",
        "six>=1.0.0",
        "oddt==0.7"

    ],
    python_requires=">3.6,<3.8",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "chemistry",
        "machine learning",
        "affinity prediction",
        "message passing neural network",
        "graph neural network",
    ],
    
)
