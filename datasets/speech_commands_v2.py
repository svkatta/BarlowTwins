import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os

from datasets.data_utils import DataUtils

class SpeechCommandsV2(Dataset):
    def __init__(self, type ,
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):
        self.root_dir = "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/speechv2/train/"
        if type == "train" :
            annotations_file=os.path.join(self.root_dir,"train_data.csv")
        elif type == "test":
            annotations_file=os.path.join(self.root_dir,"test_data.csv")    
        else:
            raise NotImplementedError
        self.uttr_df= pd.read_csv(annotations_file)
        self.transform = transform
        self.sample_rate = sample_rate
    
        self.labels = ['unknown', 'down', 'go', 'silence', 'on', 'stop', 'left', 'no','up', 'yes', 'off', 'right']
        # [ "bird","down","follow","go","learn", 
        #                 "nine","on","seven","stop","two","wow","backward","cat","eight","forward","happy",
        #                 "left","no","one","sheila","three","up", "yes","bed","dog","five",
        #                 "four","house","marvin","off","right","six","tree","visual","zero"]
        self.no_of_classes=len(self.labels) # 35

    def __len__(self):
        return len(self.uttr_df)

    def get_label_id(self,label):
        try:
            label_id = self.labels.index(label)
        except ValueError as e:
            raise NotImplementedError  
        return label_id 

    def __getitem__(self, idx):
        row = self.uttr_df.iloc[idx,:]
        uttr_path =os.path.join(self.root_dir,row['Path'])
        uttr_melspec = np.load(uttr_path)
        label = row['Label']
        return uttr_melspec, self.get_label_id(label)

