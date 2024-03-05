import numpy as np

from numpy.core.arrayprint import dtype_short_repr
import pandas as pd

import torch.utils.data as data


class userDataset(data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        #self.friends = np.loadtxt(cfg['user_path'], dtype=np.int64, skiprows=1567910, max_rows=1114)
        #self.friends = np.genfromtxt(cfg['user_path'], dtype=int, skip_header=1567910, max_rows=1114)
        self.friends = pd.read_csv(cfg['user_path'], dtype=np.int64, skiprows=567910, nrows=1114).to_numpy()
        #df = pd.DataFrame(self.friends).to_numpy()
        # self.friends = self.friends[1547278:1562650]


    def __len__(self):
        return self.friends.shape[0]

    def __getitem__(self, index: int):
        return self.friends[index, :]
