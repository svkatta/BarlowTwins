import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os

from datasets.data_utils import DataUtils

class TutUrbanSounds(Dataset):
    '''
    audio_root: /speech/Databases/Birdsong/TutUrban/TUT-urban-acoustic-scenes-2018-development/audio
    '''
    def __init__(self, type ,
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):
        audio_root = DataUtils.root_dir["tut_urban"]
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
        self.labels_dict = {'airport': 0, 'bus': 1, 'metro': 2, 'metro_station': 3, 'park': 4,
         'public_square': 5, 'shopping_mall': 6, 'street_pedestrian': 7,
         'street_traffic': 8, 'tram': 9}
        self.no_of_classes= len(self.labels_dict)

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path = os.path.join(self.audio_root,row['Path'])
        uttr_melspec = np.load(uttr_path)
        label = row['Label_id']
        return uttr_melspec, label

