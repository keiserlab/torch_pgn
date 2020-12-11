"""Helper functions for the model construction functions."""
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch.nn import LogSoftmax, Softmax

def get_sparse_fnctn(sparse_type: str):
    if sparse_type == 'log_softmax':
        return LogSoftmax(dim=0)
    elif sparse_type == 'softmax':
        return Softmax(dim=0)
    elif sparse_type == 'none':
        return lambda x: x
    else:
        # TODO: add better error parsing/catching
        return Exception("{0} is an invalid sparse_type. Please choose either 'log_softmax' or 'softmax'.".
                         format(sparse_type))


def get_pool_function(pool_type: str):
    """Returns a fuctional form of the proper pooling function given input"""
    if pool_type == 'add':
        return lambda x, batch: global_add_pool(x.squeeze(0), batch)
    elif pool_type == 'mean':
        return lambda x, batch: global_mean_pool(x.squeeze(0), batch)
    elif pool_type == 'max':
        return lambda x, batch: global_add_pool(x.squeeze(0), batch)
    else:
        # TODO: add better error parsing/catching
        return Exception("{0} is an invalid pool_type. Please choose between 'add', 'mean' or 'max' pooling.".
                         format(pool_type))