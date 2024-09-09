from torch_pgn.train.Trainer import Trainer
from torch_pgn.args import TrainArgs

def run_training(args):
    """
    Wrapper for running training given TrainArgs
    :param args: TrainArgs object with parameters to be used in training
    :return: torch_pgn.train.Trainer.Trainer object
    """
    trainer = Trainer(args)
    trainer.load_data()
    trainer.run_training()
    print(trainer.valid_eval)
    return trainer

def train():
    """Processes training arguements and runs training using the specified parameters/

    This serves as an entry points for the command line 'train' command.
    """
    args = TrainArgs()
    args.parse_args()
    args.process_args()
    run_training(args)