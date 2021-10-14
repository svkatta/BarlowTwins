import torch
# import torchaudio
# import librosa
# import tensorflow as tf
import numpy as np

def collate_fn_padd_2b(cls,batch):
    '''
    Padds batch of variable length
    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## padd
    
    batch_x = [torch.Tensor(t) for t,y in batch]
    batch_y = [torch.Tensor(y) for t,y in batch]
    batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,batch_first = True)
    batch_x = batch_x.unsqueeze(1)
    batch_y = torch.nn.utils.rnn.pad_sequence(batch_y,batch_first = True)
    batch_y = batch_y.unsqueeze(1)
    return batch_x,batch_y