"""Class allowing for better multiple training runs."""

from pgn.train.train_model import train_model

class Trainer():
    """Class that loaders and holds the arguments and data objects. Allows for easy evaluation and retraining when the
    same data is going to be used multiple times."""
    def __init__(self, args):
        #Need to see if this is going to work with subparsers
        self.args = args
