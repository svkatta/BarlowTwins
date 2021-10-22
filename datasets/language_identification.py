import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os

from datasets.data_utils import DataUtils

class LanguageIdentification(Dataset):
    '''
    audio_root: /
    '''
    def __init__(self, type ,
                    transform=None,
                    sample_rate=16000):
        audio_root = DataUtils.root_dir["language_identification"]
        if(type == "train"):
            annotations_file = os.path.join(audio_root,"train_data.csv") 
        elif(type=="test"):
            annotations_file = os.path.join(audio_root,"test_data.csv")    
        else:
            raise NotImplementedError    
        self.uttr_labels= pd.read_csv(annotations_file)
        self.audio_root = audio_root
        self.transform = transform
        self.sample_rate = sample_rate
        self.labels_dict = {'french':0, 'spanish':1, 'german':2, 'russian':3, 'english':4, 'italian':5}
        self.no_of_classes= len(self.labels_dict)

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path = os.path.join(self.audio_root,row['Path'])
        uttr_melspec = np.load(uttr_path)
        label = row['Label_id']
        return uttr_melspec, label

