from tqdm import tqdm
from pgn.train.train_utils import format_batch
import torch.nn.functional as F


def train(model, data_loader, loss_function, optimizer, scheduler,
          train_args, epoch_num = 0, logger = None, writer = None, device='cpu'):
    """
    Trains the model for an epoch.
    :param model: A PFPNetwork to be trained
    :param data_loader: The Dataloader to be used for the epoch of training
    :param loss_function: The function used to calculate the loss of the model
    :param optimizer: The optimizer used for training
    :param scheduler: A learning rate scheduler
    :param train_args: The arguments used to determine the training method
    :param epoch_num: The epoch number of training
    :param logger: A tensorboard logger used to record details of training
    :param writer: A tensorboard writer used to output recorded training details
    :return: The average loss of the epoch.
    """
    debug = logger.debug if logger is not None else print

    model.train()
    total_loss = 0

    for batch in tqdm(data_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(format_batch(train_args, batch)), batch.y, batch.num_graphs)
        loss.backward()
        total_loss += loss.item() * batch.num_graphs
        optimizer.step()

    return total_loss / len(data_loader.dataset)


