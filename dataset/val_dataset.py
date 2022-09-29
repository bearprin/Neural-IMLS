import torch
import torch.utils.data as data

from utility import get_query_point


class ValDataset(data.Dataset):
    def __init__(self, bd=0.55, resolution=128):
        super(ValDataset, self).__init__()
        self.p = get_query_point(bd=bd, resolution=resolution)

    def __len__(self):
        return self.p.shape[0]

    def __getitem__(self, index):
        return self.p[index]
