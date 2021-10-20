import argparse
import numpy as np
import logging
from pathlib import Path

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_upstream_parser():
    parser = argparse.ArgumentParser(description='Barlow Twins Training')
    # parser.add_argument('data', type=Path, metavar='DIR',
    #                     help='path to dataset')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                        help='base learning rate for weights')
    parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                        help='base learning rate for biases and batch norm parameters')
    parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                        help='print frequency')
    parser.add_argument('--exp-dir',default='./exp/',type=Path,help="experiment root directory")
    parser.add_argument('--checkpoint-file', default=None, type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--resume', default='./checkpoint/', type=str2bool,
                        metavar='DIR', help='path to checkpoint file')
    parser.add_argument('--final_pooling_type', default='Max', type=str,
                        help='valid final pooling types are Avg,Max')
    return parser
    # parser.add_argument('--projector', default='8192-8192-8192', type=str,
    #                     metavar='MLP', help='projector MLP')

def get_downstream_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--down_stream_task', default="iemocap", type=str,
                        help='''down_stream task name one of 
                        birdsong_freefield1010 , birdsong_warblr ,
                        speech_commands_v1 , speech_commands_v2
                        libri_100 , musical_instruments , iemocap , tut_urban , voxceleb1 , musan
                        ''')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size ')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--resume', default = False, type=str2bool,
                        help='number of total epochs to run')
    parser.add_argument('--pretrain_path', default=None, type=Path,
                        help='Path to Pretrain weights') 
    parser.add_argument('--freeze_effnet', default=True, type=str2bool,
                        help='Path to Pretrain weights')  
    parser.add_argument('--final_pooling_type', default='Avg', type=str,
                        help='valid final pooling types are Avg,Max')                                                            
    parser.add_argument('--load_only_efficientNet',default = True,type =str2bool)  
    parser.add_argument('--tag',default = "pretrain_big",type =str)
    parser.add_argument('--exp-dir',default='./exp/',type=Path,help="experiment root directory")    
    parser.add_argument('--lr',default=0.001,type=float,help="experiment root directory")                    
    return parser

def freeze_effnet(model):
    logger=logging.getLogger("__main__")
    logger.info("freezing effnet weights")
    for param in model.parameters():
        param.requires_grad = False
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Metric(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        if isinstance(val, (torch.Tensor)):
            val = val.numpy()
            self.val = val
            self.sum += np.sum(val) 
            self.count += np.size(val)
        self.avg = self.sum / self.count

def move_to_gpu(gpu,*args):
    if torch.cuda.is_available():
        for item in args:
            item.cuda(gpu)


def load_pretrain(path,model,
                load_only_effnet=False,freeze_effnet=False):
    logger=logging.getLogger("__main__")
    logger.info("loading from checkpoint only weights : "+path)
    checkpoint = torch.load(path)
    if load_only_effnet :
        for key in checkpoint['state_dict'].copy():
            if not key.startswith('backbone'):
                del checkpoint['state_dict'][key]
    mod_missing_keys,mod_unexpected_keys   = model.load_state_dict(checkpoint['state_dict'],strict=False)
    assert mod_missing_keys == ['fc.weight', 'fc.bias'] and mod_unexpected_keys == []
    return model
##------------------------------------------------##
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import random
import torchvision.transforms as transforms

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

