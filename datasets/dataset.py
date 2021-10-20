from datasets.language_identification import LanguageIdentification
from datasets.voxceleb import Voxceleb1
from datasets.birdsong_dataset import BirdSongDataset 
from datasets.libri100 import Libri100 
from datasets.tut_urban_sounds import TutUrbanSounds 
from datasets.musical_instruments import MusicalInstruments
from datasets.iemocap import IEMOCAPDataset 
from datasets.speech_commands_v1 import SpeechCommandsV1 
from datasets.speech_commands_v2 import SpeechCommandsV2 
import torch

def get_dataset(downstream_task_name):
    if downstream_task_name == "birdsong_combined":
        return split_dataset(BirdSongDataset(type="combined"))         
    elif downstream_task_name == "speech_commands_v1":
        return SpeechCommandsV1(type="train") , SpeechCommandsV1(type="test")
    elif downstream_task_name == "speech_commands_v2":
        return SpeechCommandsV2(type="train") , SpeechCommandsV2(type="test")      
    elif downstream_task_name == "libri_100":
        return Libri100(type="train") , Libri100(type="test")      
    elif downstream_task_name == "musical_instruments":
        return MusicalInstruments(type="train") , MusicalInstruments(type="test")
    elif downstream_task_name == "iemocap":
        return IEMOCAPDataset(type='train'),IEMOCAPDataset(type='test')            
    elif downstream_task_name == "tut_urban": 
        return TutUrbanSounds(type="train"),TutUrbanSounds(type="test")    
    elif downstream_task_name == "voxceleb_v1":
        return Voxceleb1(type="train") , Voxceleb1(type="test")       
    elif downstream_task_name == "language_identification":
        return LanguageIdentification(type="train"), LanguageIdentification(type="test")                 
    else:
        raise NotImplementedError


def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataset.no_of_classes = dataset.no_of_classes
    return train_dataset,valid_dataset