from tqdm import tqdm
from pgn.train.train_utils import _format_batch


def train(model, data_loader, loss_function, optimizer, scheduler,
          train_args, epoch_num = 0, logger = None, writer = None):
    """
    Trains the model for an epoch.

    TODO: docstring
    """
    debug = logger.debug if logger is not None else print

    model.train()
    total_loss = 0

    device = model.device

    for batch in tqdm(data_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = loss_function(train_args, batch)
        loss.bachward()
        total_loss += loss.item() * batch.num_graphs
        optimizer.step()

    return total_loss / len(data_loader.dataset)


