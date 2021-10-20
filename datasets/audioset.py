import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os
from datasets.specaugment import specaug

class AudiosetDataset(Dataset):
    def __init__(self, 
                    transform=None,
                    sample_rate=16000):        
        self.feat_root = "/speech/srayan/icassp/kaggle_data/audioset_train/spec/" 
        annotations_file=os.path.join("/speech/sandesh/icassp/aaai/datasets/audioset.csv")    
        self.uttr_labels= pd.read_csv(annotations_file)
        self.transform = transform
        self.sample_rate = sample_rate
        

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        path = self.uttr_labels.iloc[idx,:]['path']
        uttr_path = os.path.join(self.feat_root,path)
        clean_uttr =  torch.tensor(np.load(uttr_path).tolist())
        specaug_uttr = specaug(clean_uttr.clone().detach())
        return clean_uttr, specaug_uttr

