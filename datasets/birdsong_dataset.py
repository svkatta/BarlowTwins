import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os

from datasets.data_utils import DataUtils

class BirdSong(Dataset):
    def __init__(self, 
                    type, 
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):
        
        self.feat_root =  DataUtils.root_dir["Birdsong"]
        if type == "freefield1010":
            annotations_file=os.path.join(self.feat_root,"freefield1010_data.csv")
        elif type == "Warblr":
            annotations_file=os.path.join(self.feat_root,"Warblr_data.csv")
        elif type == "combined":
            annotations_file=os.path.join(self.feat_root,"combined_data.csv")
        else :
            raise NotImplementedError    
        self.uttr_labels= pd.read_csv(annotations_file)
        
        self.transform = transform
        self.sample_rate = sample_rate
        self.no_of_classes=2

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['Path'])
        uttr_melspec = np.load(uttr_path)
        label = row['Label']
        return uttr_melspec, label


class BirdSongDatasetL2(Dataset):
    def __init__(self, 
                    type, 
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):
        
        self.feat_root =  DataUtils.root_dir["Birdsong"]
        if type == "freefield1010":
            annotations_file=os.path.join(self.feat_root,"freefield1010_data.csv")
        elif type == "Warblr":
            annotations_file=os.path.join(self.feat_root,"Warblr_data.csv")
        elif type == "combined":
            annotations_file=os.path.join(self.feat_root,"combined_data.csv")
        else :
            raise NotImplementedError    
        self.uttr_labels= pd.read_csv(annotations_file)
        
        self.transform = transform
        self.sample_rate = sample_rate
        self.no_of_classes=2

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['PathL2'])
        uttr_melspec = np.load(uttr_path)
        label = row['Label']
        return uttr_melspec, label