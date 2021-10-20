import torch
# import torchaudio
# import librosa
# import tensorflow as tf
import numpy as np

# def collate_fn_padd_2b(batch):
#     '''
#     Padds batch of variable length
#     note: it converts things ToTensor manually here since the ToTensor transform
#     assume it takes in images rather than arbitrary tensors.
#     '''
#     ## padd
    
#     batch_x = [torch.Tensor(t) for t,y in batch]
#     batch_y = [torch.Tensor(y) for t,y in batch]
#     batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,batch_first = True)
#     batch_x = batch_x.unsqueeze(1)
#     batch_y = torch.nn.utils.rnn.pad_sequence(batch_y,batch_first = True)
#     batch_y = batch_y.unsqueeze(1)
#     return batch_x,batch_y


def collate_fn_padd_downstream(batch):
    '''
    Padds batch of variable length
    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    batch_x = [torch.Tensor(t) for t,y in batch]
    batch_y = [y for t,y in batch]
    batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,batch_first = True)
    batch_x = batch_x.unsqueeze(1)
    batch_y = torch.Tensor(batch_y).type(torch.LongTensor)
    return batch_x,batch_y

#-------------
class DataUtils():

    root_dir ={
        "Birdsong" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/birdsong",
        "iemocap" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/iemocap/",
        "MusicalInstruments" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/magenta",
        "tut_urban" : "/speech/Databases/Birdsong/TutUrban/TUT-urban-acoustic-scenes-2018-development",
        "voxceleb_v1" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/voxceleb/",
        "language_identification" :"/speech/Databases/Birdsong/lid",
        "libri_100" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/libri100",
        "speech_commands_v1" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/SpeechCommandsV1",
        "speech_commands_v2" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/SpeechCommandsV2"
    }

#-------------Transforms Audio and Spectrogram-----------------
