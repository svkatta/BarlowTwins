import json
import math
import os
import logging
import signal
import subprocess
import sys
import time
import wandb
wandb.login()

from torch import nn
import torch
import torchvision
from datasets.audioset import AudiosetDataset
from efficientnet.model import BarlowTwins
from optmizers.lars import LARS , adjust_learning_rate

from datasets.data_utils import collate_fn_padd_2b
from utils import get_upstream_parser ,AverageMeter


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass



def main_worker(gpu, args):
    args.rank += gpu
    if args.rank==0:
        run = wandb.init(project="pytorch-demo", config=vars(args))
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    stats_file=None
    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    
    if args.resume:
        ckpt = torch.load(args.checkpoint_file,map_location='cpu')
        print("Resuming pretrain from epoch {0}".format(ckpt['epoch']))
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    
    dataset = AudiosetDataset() # ! rewrite this 
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler,collate_fn = collate_fn_padd_2b)

    scaler = torch.cuda.amp.GradScaler()
    if args.rank == 0:
        print("Starting To Train")
        wandb.watch(model, log="all", log_freq=10)
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        train_one_epoch(epoch,model,optimizer,loader,scaler,args,gpu,stats_file)
        
        # if args.rank == 0:
            # save checkpoint            
            # torch.save({'epoch': epoch + 1,
            #             'model': model.state_dict(),
            #             'optimizer' : optimizer.state_dict()},
            #         args.exp_dir / 'checkpoints' / ('checkpoint_' + str(epoch + 1)  + '.pth'))
    if args.rank==0:
        run.finish()

def train_one_epoch(epoch,model,optimizer,loader,scaler,args,gpu,stats_file):
    # per epoch stats
    batch_time = AverageMeter()
    losses = AverageMeter()
    on_diag_losses = AverageMeter()
    off_diag_losses = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    for step, (y1, y2) in enumerate(loader,start=epoch * len(loader)): 
        data_time.update(time.time() - end)

        y1 = y1.cuda(gpu, non_blocking=True)
        y2 = y2.cuda(gpu, non_blocking=True)
        adjust_learning_rate(args, optimizer, loader, step)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss,on_diag,off_diag = model.forward(y1, y2)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        
        losses.update(loss, y1.size(0))
        on_diag_losses.update(on_diag,y1.size(0))
        off_diag_losses.update(off_diag,y1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0:
            wandb.log({"epoch": epoch, "instant_loss": loss}, step=step)
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                    .format(epoch, step%len(loader), len(loader), batch_time=batch_time,data_time=data_time, loss=losses))

        if step % args.print_freq == 0:
            if args.rank == 0:
                wandb.log({"lr_weights" :optimizer.param_groups[0]['lr'] ,
                            "lr_biases" :optimizer.param_groups[1]['lr'],
                            }, step=step)
    if args.rank == 0:
        wandb.log({"loss_epoch":losses.avg , "on_diag_loss": on_diag_losses.avg,
                    "off_diag_loss" : off_diag_losses.avg },step=(epoch+1)*len(loader))
    return losses.avg



def main():
    parser=get_upstream_parser()
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    
    # single-node distributed training
    args.rank = 0
    args.dist_url = 'tcp://localhost:58362'
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


if __name__ == '__main__':
    main()
