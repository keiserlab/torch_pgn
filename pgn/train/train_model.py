from pgn.train.train import train
from pgn.train.evaluate_model import evaluate
from pgn.train.train_utils import parse_loss, make_save_directories, save_checkpoint, load_checkpoint
from pgn.models.model import PFPNetwork

import os.path as osp

from tqdm import trange

from torch_geometric.data import DataLoader
import torch


def train_model(args, train_data, validation_data, test_data=None):
    """
    Function to run a complete run of training. The function also constructs the model and writes the output of the
    training to the specified model directory (see documentation).
    :param args: The TrainArgs container that contains the training parameters and settings
    :param train_data: The training data as a ProximityGraphDataset
    :param validation_data: The validation data as a ProximityGraphDatatset
    :param test_data: The testing data as a ProximityGraphDataset (defaults to None for training instances where the
    held out test-set is not used for evaluation)
    :return: The best best model from training as determined by validation score and a dictionary of the metrics to
    evaluate validation performance.
    """
    #TODO: add logging
    torch_seed = args.torch_seed
    loss_fucntion = parse_loss(args)
    num_workers = args.num_workers

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size,
                                  num_workers=num_workers, shuffle=True, seed=args.seed)
    valid_dataloader = DataLoader(validation_data, batch_size=args.batch_size,
                                  num_workers=num_workers, shuffle=False)
    if args.load_test is True:
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size,
                                  num_workers=num_workers, shuffle=False)

    save_dir = args.save_dir
    # Format the save_dir for the output of training data
    make_save_directories(save_dir)

    #TODO: Add function for loading from a prexisting checkpoint
    #TODO: Fix dimension argument (either remove or calculate from the data above).
    model = PFPNetwork(args, args.node_dim, args.edge_dim)

    model = model.to(args.device)

    #TODO: Add options to allow for optimizer choice
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    #TODO: Allow for more sophisticated schedular choices
    if args.weight_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.7, patience=10,
                                                               min_lr=1e-6)
    else:
        scheduler = None

    best_score = float('inf')
    best_epoch = 0
    for epoch in trange(args.epochs):
        train_loss = train(model=model,
                           data_loader=train_dataloader,
                           loss_function=loss_fucntion,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           train_args=args,
                           epoch_num=epoch)

        validation_eval = evaluate(model=model,
                                   data_loader=valid_dataloader,
                                   args=args,
                                   metrics=args.metrics,
                                   mean=args.label_mean,
                                   std=args.label_std)


        validation_loss = validation_eval[args.loss_function]
        if validation_loss < best_score:
            best_score = validation_loss
            best_epoch = epoch
            save_checkpoint(osp.join(save_dir, 'best_checkpoint.pt'),
                            model=model,
                            args=args)

    model = load_checkpoint(osp.join(save_dir, 'best_checkpoint.pt'),
                            device=args.device)

    validation_eval = evaluate(model=model,
                               data_loader=valid_dataloader,
                               args=args,
                               metrics=args.metrics,
                               mean=args.label_mean,
                               std=args.label_std)

    return model, validation_eval




