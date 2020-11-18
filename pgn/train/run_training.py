from pgn.train.Trainer import Trainer

def run_training(args):
    """
    Wrapper for running training given TrainArgs
    :param args: TrainArgs object with parameters to be used in training
    :return: pgn.train.Trainer.Trainer object
    """
    trainer = Trainer(args)
    trainer.load_data()
    trainer.run_training()
    return trainer