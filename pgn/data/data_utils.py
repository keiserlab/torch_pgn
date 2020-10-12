import numpy as np
import torch

class OneHotTransform(object):
    """
    Transform object for ProximityGraphDataset with atomic number feature. Transforms the int feature into a 1-hot
    encoding that used atomic number (up to 100) as an index.
    #TODO: make it compatible with other atomic number index positions.
    """
    def __call__(self, data, encoder=None):
        device = data.x.device
        atom_type = data.x[:, 0].numpy().astype(int)
        one_hots = np.zeros((atom_type.shape[0], 100))
        for i, row in enumerate(one_hots):
            one_hots[i, atom_type[i]] += 1
        data.x = np.hstack((one_hots, data.x[:, 1:]))
        data.x = torch.from_numpy(data.x).type(torch.FloatTensor).to(device)
        return data


def normalize_targets(dataset, mean=None, std=None, yield_stats=False):
    """
    Normalizes the training target to have mean 0 and stddev 1
    :param dataset: dataset to normalize the targets for
    :param mean: external mean to use for normalization (i.e. test set normalization)
    :param std: external stddev to use for normalization (i.e. test set normalization)
    :param yield_stats: toggle to yield [dataset, (mean, std)] if True or just [dataset] if False
    :return: A dataset with normalized targets
    """
    if mean is None:
        mean = dataset.data.y.mean()
    if std is None:
        std = dataset.data.y.std()

    dataset.data.y = (dataset.data.y - mean) / std

    if yield_stats:
        return [dataset]
    else:
        return [dataset, (mean, std)]