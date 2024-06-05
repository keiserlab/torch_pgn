"""Helper functions for the model construction functions."""
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch.nn import LogSoftmax, Softmax

def get_sparse_fnctn(sparse_type: str):
    if sparse_type == 'log_softmax':
        return LogSoftmax(dim=-1)
    elif sparse_type == 'softmax':
        return Softmax(dim=-1)
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


# From chemprop
def index_select_ND(source, index):
    """
    Selects the message features from source corresponding to the atom or bond indices in :code:`index`.
    :param source: A tensor of shape :code:`(num_bonds, hidden_size)` containing message features.
    :param index: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds)` containing the atom or bond
                  indices to select from :code:`source`.
    :return: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds, hidden_size)` containing the message
             features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target