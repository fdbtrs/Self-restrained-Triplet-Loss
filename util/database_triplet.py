import pickle
from typing import Tuple, Callable, List

from sklearn import preprocessing
import torch
import os
import numpy as np
import cv2



class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, root: str ='/home/aboller/ArcFace/ms1m_features_dlib/', random: bool = False, isTraining: bool= False, emb_size=128) -> None:
        self.data_dir = root
        self.random=random
        # scan the data directory for all files
        self.files, self.classes, self.dirs,self.uniqe_class = self.scan(self.data_dir)
        self.label=1
        self.isTraining=isTraining
        self.emb_size=emb_size

    def scan(self, dir) -> Tuple[List[str], List[str]]:
        input_dir = os.path.join(dir, 'fakemask')
        files = []
        classes = []
        dirs=[]
        uniqe_class=[]
        cls=0
        lst_dir=os.listdir(input_dir)
        for c in range(0,len(lst_dir)):
            for f in os.listdir(os.path.join(input_dir, lst_dir[c])):
                files.append(f)
                classes.append(cls)
                dirs.append(lst_dir[c])
            cls=cls+1
        return files, classes,dirs,uniqe_class

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
        c = self.dirs[idx]
        cls = self.classes[idx]
        #Get positive pair
        #unmask_embedding = np.load(os.path.join(self.data_dir, 'original', str(c), f))
        face_embedding = np.load(os.path.join(self.data_dir, 'original', str(c), f))
        mask_embedding = np.load(os.path.join(self.data_dir, 'fakemask', str(c), f))
        while(True):
             radom_class=np.random.choice(range(-40,40))
             if((idx+radom_class)<len(self)):
              if (cls !=  self.classes[radom_class]):
                  radom_index=radom_class
                  negative_embedding = np.load(os.path.join(self.data_dir, 'fakemask', str(self.dirs[radom_index]), self.files[radom_index]))
                  break
        mask_embedding= mask_embedding.flatten()
        face_embedding=face_embedding.flatten()
        negative_embedding=negative_embedding.flatten()
        #unmask_embedding=unmask_embedding.flatten()

        mask_embedding = torch.tensor(mask_embedding )
        face_embedding = torch.tensor(face_embedding )
        negative_embedding = torch.tensor(negative_embedding )
        #unmask_embedding=torch.tensor(unmask_embedding)
        #return (mask_embedding,face_embedding,negative_embedding,unmask_embedding,self.label,f)

        return (mask_embedding, face_embedding, negative_embedding, self.label, f)





