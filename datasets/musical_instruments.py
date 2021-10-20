import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os

from datasets.data_utils import DataUtils



class MusicalInstruments(Dataset):
    def __init__(self,  
                    type,
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):        
        self.feat_root =  DataUtils.root_dir["MusicalInstruments"]
        if(type=="train"):
            annotations_file = os.path.join(self.feat_root,'train_data.csv')
        elif(type=="valid"):
            annotations_file = os.path.join(self.feat_root,'valid_data.csv')
        elif(type=="test"):
            annotations_file = os.path.join(self.feat_root,'test_data.csv')    
        else:
            raise NotImplementedError

        self.uttr_labels= pd.read_csv(annotations_file)
        self.transform = transform
        self.sample_rate = sample_rate
        self.no_of_classes= 11

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        path,label = self.uttr_labels.iloc[idx,:]
        uttr_path = os.path.join(self.feat_root,path)
        uttr_melspec = np.load(uttr_path)
        return uttr_melspec, label

