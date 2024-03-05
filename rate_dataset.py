import numpy as np
import pandas as pd

import torch.utils.data as data


class rateDataset(data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        #self.votes = np.loadtxt(cfg['rating_path'], dtype=np.int64, skiprows=299999, max_rows=20000)
        #self.votes = np.loadtxt(cfg['rating_path'], skiprows=299999, usecols=range(3), max_rows=20000)
        #self.votes = np.genfromtxt(cfg['rating_path'], dtype=int, skip_header=299999, max_rows=20000)
        self.votes = pd.read_csv(cfg['rating_path'], dtype=np.int64, skiprows=299999, nrows=20000).to_numpy()
        #df = pd.DataFrame(self.votes).to_numpy()
        # self.votes = self.votes[0:200000]

        
    def __len__(self):
        return self.votes.shape[0]

    def __getitem__(self, index: int):
        return self.votes[index, :]
