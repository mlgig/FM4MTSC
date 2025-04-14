import random
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class TimeSeriesDatasetLazy(Dataset):
    def __init__(self, data_root, tasks, split='TEST', sequence_length=512, 
                 balance=False):
        # === Balanced ===
        if balance:
            dataset_dirs = []
            for task in tasks:
                dataset_dirs.extend(glob.glob(f"{data_root}/{task}/{split}/*"))
            files = [glob.glob(f"{x}/*.csv") for x in dataset_dirs]
            data = []
            if split == "TRAIN":
                max_dataset_samples = max([len(x) for x in files])
                for dataset in files:
                    if len(dataset) < max_dataset_samples:
                        data.extend(dataset * np.ceil(max_dataset_samples/ len(dataset)).astype(int))
                    else:
                        data.extend(dataset)
            else:
                data = glob.glob(f"{data_root}/{split}/*/*.csv")
                
            random.shuffle(data)
            self.data = np.array(data)
        else:
            # === Original ===
            data = []
            for task in tasks:
                data.extend(glob.glob(f"{data_root}/{task}/{split}/*/*.csv"))
            random.shuffle(data)
            self.data = np.array(data)

        self.sequence_length = sequence_length

    def __getitem__(self, idx):
        x = torch.from_numpy(pd.read_csv(self.data[idx]).values.flatten()).float().view(1, -1)
        if not x.shape[-1] == self.sequence_length:
            x = F.interpolate(x.view(1, 1, -1), self.sequence_length, mode='linear')
        x = (x - x.mean(-1, keepdims=True)) / (x.var(-1, keepdims=True) + 1e-5).sqrt()
        x = x.squeeze()
        return x 

    def __len__(self):
        return len(self.data)
