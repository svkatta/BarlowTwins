import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os

from datasets.data_utils import DataUtils

class IEMOCAPDataset(Dataset):
    def __init__(self, type, 
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):        
        self.feat_root =  DataUtils.root_dir["iemocap"]
        if type == "train":
            annotations_file=os.path.join(self.feat_root,"train_data.csv")
        elif type == "test":
            annotations_file=os.path.join(self.feat_root,"test_data.csv")    
        else:
            raise NotImplementedError
        self.uttr_labels= pd.read_csv(annotations_file)
        self.transform = transform
        self.sample_rate = sample_rate
        self.labels_dict ={'neu':0, 'ang':1, 'sad':2, 'hap':3} #{'ang': 0, 'dis': 1, 'exc': 2, 'fea': 3, 'fru': 4, 'hap': 5, 'neu': 6, 'oth': 7, 'sad': 8, 'sur': 9}
        #DataUtils.map_labels(self.uttr_labels['Label'].to_numpy())
        self.no_of_classes= len(self.labels_dict)

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['Path'])
        uttr_melspec = np.load(uttr_path)
        label = row['Label']
        return uttr_melspec, self.labels_dict[label]

