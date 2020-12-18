from pgn.train.train import train
from pgn.train.evaluate_model import evaluate
from pgn.train.train_utils import parse_loss, make_save_directories, save_checkpoint, load_checkpoint
from pgn.models.model import PFPNetwork
from pgn.evaluate.plot_utils import plot_correlation

from torch.utils.tensorboard import SummaryWriter

import os.path as osp
import copy

from tqdm import trange

from torch_geometric.data import DataLoader
import torch
import torch.nn.functional as F


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
    model = PFPNetwork(args, args.num_node_feature, args.num_edge_features)

    torch.manual_seed(0)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    device = args.device

    save_dir = args.save_dir
    # Format the save_dir for the output of training data
    # make_save_directories(save_dir)

    #TODO: Add function for loading from a prexisting checkpoint
    #TODO: Fix dimension argument (either remove or calculate from the data above).

    model = model.to(args.device)

    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    if args.weight_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.7, patience=10,
                                                               min_lr=1e-6)
    best_val_error = float('inf')
    best_params = None
    model.to(args.device)

    writer = SummaryWriter(log_dir=args.save_dir)
    for epoch in range(1, args.epochs):
        if args.weight_decay and epoch > 20:
            lr = scheduler.optimizer.param_groups[0]['lr']

        loss = train(model, train_loader, F.mse_loss, optimizer, scheduler, args, device=args.device)

        metrics = evaluate(model, val_loader, args, args.metrics)
        val_error = metrics['mse']
        if val_error < best_val_error:
            best_val_error = val_error
            best_params = copy.deepcopy(model.state_dict())
        if args.weight_decay and epoch > 20:
            scheduler.step(val_error)
        print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}'.format(epoch, lr, loss, val_error))
        writer.add_scalar(f'train loss_mse', loss, epoch + 1)
        for metric, value in metrics.items():
            writer.add_scalar(f'validation_{metric}', value, epoch + 1)
    model.load_state_dict(best_params)
    save_checkpoint(osp.join(save_dir, 'best_checkpoint.pt'),
                    model=model,
                    args=args)