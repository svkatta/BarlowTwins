import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os

from datasets.data_utils import DataUtils

# stats
# train total : 064722
# test  total : 158539

class SpeechCommandsV1(Dataset):
    def __init__(self, annotations_file="/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/SpeechCommandsV1/train/train_data.csv",
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):
        self.uttr_df= pd.read_csv(annotations_file)
        self.transform = transform
        self.sample_rate = sample_rate
        
        test_labels = ["yes", "no", "up", "down","left", "right", "on", "off", "stop", "go" ] #10
        core_labels = test_labels + ["zero","one","two","three","four","five","six","seven","eight","nine"] # 20
        auxliary_labels = [] # ["bird","dog","happy", "wow","bed","cat","house","marvin","sheila","tree"] #30
        self.labels = core_labels + auxliary_labels
        self.no_of_classes=len(self.labels) # 30

    def __len__(self):
        return len(self.uttr_df)

    def get_label_id(self,label):
        try:
            label_id = self.labels.index(label)
        except ValueError as e:
            label_id = len(self.labels)  
        return label_id 

    def __getitem__(self, idx):
        audio_path,label = self.uttr_df.iloc[idx,:]
        uttr_melspec = np.load(audio_path+'.npy')
        return uttr_melspec, self.get_label_id(label)

