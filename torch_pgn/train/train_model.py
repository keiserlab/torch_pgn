from torch_pgn.train.train import train
from torch_pgn.train.evaluate_model import evaluate
from torch_pgn.train.train_utils import parse_loss, make_save_directories, save_checkpoint, load_checkpoint
from torch_pgn.models.model import PGNNetwork
from torch_pgn.evaluate.plot_utils import plot_correlation

from torch.utils.tensorboard import SummaryWriter



import os.path as osp

from tqdm import trange

from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
import torch


def train_model(args, train_data, validation_data, test_data=None):
    """
    Function to run a complete run of training. The function also constructs the model and writes the output of the
    training to the specified model directory (see documentation).
    :param args: The TrainArgs container that contains the training parameters and settings
    :param train_data: The training working_data as a ProximityGraphDataset
    :param validation_data: The validation working_data as a ProximityGraphDatatset
    :param test_data: The testing working_data as a ProximityGraphDataset (defaults to None for training instances where the
    held out test-set is not used for evaluation)
    :return: The best best model from training as determined by validation score and a dictionary of the metrics to
    evaluate validation performance.
    """
    torch_seed = args.torch_seed
    loss_fucntion = parse_loss(args)
    num_workers = args.num_workers

    if args.fine_tuning_dir is None:
        model = PGNNetwork(args, args.node_dim, args.edge_dim)
    else:
        model = load_checkpoint(args.fine_tuning_dir, args.device)

    if args.straw_model and args.encoder_type == 'dimenet++':
        # Turn of grad for all
        for param in model.encoder.parameters():
            param.requires_grad = False
        # Reactivates params for the output blocks
        for param in model.encoder.output_blocks.parameters():
            param.requires_grad = True
    elif args.straw_model and args.encoder_type != 'fp':
        for param in model.encoder.parameters():
            param.requires_grad = False

    if args.multi_gpu == True:
        model = DataParallel(model)

    torch.manual_seed(args.seed)

    if args.multi_gpu:
        train_dataloader = DataListLoader(train_data, batch_size=args.batch_size,
                                      num_workers=num_workers, shuffle=True,
                                      timeout=0)
        valid_dataloader = DataListLoader(validation_data, batch_size=args.batch_size,
                                      num_workers=num_workers, shuffle=False,
                                      timeout=0)
    else:
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size,
                                      num_workers=num_workers, shuffle=True,
                                      timeout=0)
        valid_dataloader = DataLoader(validation_data, batch_size=args.batch_size,
                                      num_workers=num_workers, shuffle=False,
                                      timeout=0)

    if args.mode == 'experiment':
        validation_name = 'validation'
    if args.mode == 'evaluate':
        validation_name = 'test'

    save_dir = args.save_dir
    # Format the save_dir for the output of training working_data
    make_save_directories(save_dir)

    if args.device != 'cpu':
        torch.cuda.set_device(args.device)
        model.cuda(args.device)

    lr = args.lr
    #TODO: Add options to allow for optimizer choice
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    #TODO: Allow for more sophisticated scheduler choices
    if args.weight_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.7, patience=args.patience,
                                                               min_lr=1e-6)
    else:
        scheduler = None

    writer = SummaryWriter(log_dir=save_dir)

    best_score = float('inf')
    best_epoch = 0
    for epoch in trange(args.epochs):
        if args.weight_decay and epoch > args.decay_delay:
            lr = scheduler.optimizer.param_groups[0]['lr']

        train_loss = train(model=model,
                           data_loader=train_dataloader,
                           loss_function=loss_fucntion,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           train_args=args,
                           epoch_num=epoch,
                           device=args.device)

        validation_eval = evaluate(model=model,
                                   data_loader=valid_dataloader,
                                   args=args,
                                   metrics=args.metrics,
                                   mean=args.label_mean,
                                   std=args.label_std,
                                   remove_norm=False)

        #print(f'Train loss_{args.loss_function} = {train_loss:.4e}')
        #print(f"{validation_name} evaluation: ", validation_eval)

        if args.weight_decay:
            writer.add_scalar(f'learning_rate', lr, epoch + 1)

        writer.add_scalar(f'train loss_{args.loss_function}', train_loss, epoch+1)
        for metric, value in validation_eval.items():
            writer.add_scalar(f'{validation_name}_{metric}', value, epoch+1)


        validation_loss = validation_eval[args.loss_function]

        if args.weight_decay and epoch > args.decay_delay:
            scheduler.step(validation_eval[args.loss_function])

        if validation_loss < best_score:
            best_score = validation_loss
            best_epoch = epoch
            save_checkpoint(osp.join(save_dir, 'best_checkpoint.pt'),
                            model=model,
                            args=args)

    model = load_checkpoint(osp.join(save_dir, 'best_checkpoint.pt'),
                            device=args.device, return_args=False)

    train_eval = evaluate(model=model,
                          data_loader=train_dataloader,
                          args=args,
                          metrics=args.metrics,
                          mean=args.label_mean,
                          std=args.label_std)

    validation_eval = evaluate(model=model,
                               data_loader=valid_dataloader,
                               args=args,
                               metrics=args.metrics,
                               mean=args.label_mean,
                               std=args.label_std)


    if args.plot_correlations:
        plot_correlation(model=model,
                         args=args,
                         data_loader=train_dataloader,
                         filename='train_correlation',
                         metrics=train_eval
                         )

        plot_correlation(model=model,
                         args=args,
                         data_loader=valid_dataloader,
                         filename=f'{validation_name}_correlation',
                         metrics=validation_eval
                         )

        return model, validation_eval


