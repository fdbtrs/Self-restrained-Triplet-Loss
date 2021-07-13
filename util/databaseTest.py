import pickle
from typing import Tuple, Callable, List
import torch
import os
import numpy as np
import cv2



class MaskDatasetTest(torch.utils.data.Dataset):
    def __init__(self, root: str ='/home/aboller/ArcFace/ms1m_features/', random: bool = False) -> None:
        self.data_dir = root
        self.random=random
        # scan the data directory for all files
        self.files = self.scan(self.data_dir)

    def scan(self, dir) -> Tuple[List[str]]:
        files=[]
        lst_dir=os.listdir(dir)
        for f in lst_dir:
            files.append(f)
        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if the given index is out of bounds, throw an IndexError
        if idx >= len(self):
            raise IndexError

        # get the respective file name and class
        f = self.files[idx]

        mask_embedding = np.load(os.path.join(self.data_dir, f))

        mask_embedding=mask_embedding.flatten()
        mask_embedding=torch.tensor(mask_embedding)
        return (mask_embedding,f,idx)

class MaskDatasetTestMFR2(torch.utils.data.Dataset):
    def __init__(self, root: str ='/home/aboller/ArcFace/ms1m_features/', random: bool = False) -> None:
        self.data_dir = root
        self.random=random
        # scan the data directory for all files
        self.files,self.dirs = self.scan(self.data_dir)

    def scan(self, dir) -> Tuple[List[str]]:
        files=[]
        dirs=[]
        lst_dir=os.listdir(dir)
        for f in lst_dir:
            lst_f=os.listdir(os.path.join(dir,f))
            for fl in lst_f:
                files.append(fl)
                dirs.append(f)
        return files,dirs

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if the given index is out of bounds, throw an IndexError
        if idx >= len(self):
            raise IndexError
        # get the respective file name and class
        f = self.files[idx]
        dir=self.dirs[idx]

        mask_embedding = np.load(os.path.join(self.data_dir,dir, f))

        mask_embedding=mask_embedding.flatten()
        mask_embedding=torch.tensor(mask_embedding)
        return (mask_embedding,f,dir)




