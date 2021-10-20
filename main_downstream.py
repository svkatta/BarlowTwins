import logging
import os
import time
import json
import matplotlib.pyplot as plt
import torch
from torch import nn
import sys
from datasets.data_utils import DataUtils
from datasets.dataset import get_dataset
from efficientnet.model import  DownstreamClassifer
from utils import (AverageMeter,Metric,freeze_effnet,get_downstream_parser,load_pretrain)#resume_from_checkpoint, save_to_checkpoint,set_seed

import wandb
wandb.login()

def main_worker(gpu, args):
    args.rank += gpu
    if args.rank==0:
        run = wandb.init(project="tut urban", config=vars(args))
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    stats_file=None
    args.exp_root = args.exp_dir / args.tag
    args.exp_root.mkdir(parents=True, exist_ok=True)
    if args.rank == 0:
        # args.exp_root = args.exp_dir / args.tag
        # args.exp_root.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_root / 'downstream_stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True # ! change it set seed 

    # train and test loaders 
    # ! user sampler and ddp 
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    train_dataset,test_dataset = get_dataset(args.down_stream_task)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)    
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=per_device_batch_size,
                                                collate_fn = DataUtils.collate_fn_padd_2,
                                                pin_memory=True,sampler = train_sampler)
    # ! not required just run things in one gpu else need to take care of reduce operations 
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=per_device_batch_size,
                                                collate_fn = DataUtils.collate_fn_padd_2,
                                                pin_memory=True)  

    # models
    model = DownstreamClassifer(no_of_classes=train_dataset.no_of_classes,
                                final_pooling_type=args.final_pooling_type).cuda(gpu)
    
    # Resume
    start_epoch =0 
    if args.resume:
        raise NotImplementedError
        # resume_from_checkpoint(args.pretrain_path,model,optimizer)
    elif args.pretrain_path:
        load_pretrain(args.pretrain_path,model,args.load_only_efficientNet,args.freeze_effnet)
    # Freeze effnet
    if args.freeze_effnet:
        freeze_effnet(model)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
    )

    if args.rank == 0:
        print("Starting To Train")
        wandb.watch(model,criterion=criterion, log="all", log_freq=10)

    # if args.rank == 0 :
    #         eval(0,model,test_loader,criterion,args,gpu,stats_file)

    for epoch in range(start_epoch,args.epochs):
        train_sampler.set_epoch(epoch)
        train_one_epoch(train_loader, model, criterion, optimizer, epoch,gpu,args)
        # save_to_checkpoint(args.down_stream_task,args.exp_root,epoch,model,optimizer)

        if args.rank == 0 :
            eval(epoch,model,test_loader,criterion,args,gpu,stats_file)
            # test_accuracy.append(eval_stats["accuracy"].avg)
            # print(eval_stats["loss"].avg.numpy())
            # print(eval_stats["accuracy"].avg)
            # print(max(test_accuracy))
            # stats = dict(epoch=epoch,
            #         Train_loss=train_stats["loss"].avg.cpu().numpy().item(),
            #         Test_Loss=(eval_stats["loss"].avg).numpy().item(),
            #         Test_Accuracy =eval_stats["accuracy"].avg,  
            #         Best_Test_Acc=max(test_accuracy))
            # print(stats)
            # print(json.dumps(stats), file=stats_file)
    if args.rank==0:
        run.finish()


def train_one_epoch(loader, model, crit, opt, epoch,gpu,args):
    '''
    Train one Epoch
    '''
    logger = logging.getLogger(__name__)
    logger.debug("epoch:"+str(epoch) +" Started")
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    accuracy = Metric()

    model.train() # ! imp 
    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        output = model(input_tensor.cuda(gpu, non_blocking=True))
        loss = crit(output, target.cuda(gpu, non_blocking=True))
        
        losses.update(loss, input_tensor.size(0))
        opt.zero_grad()
        loss.backward()
        opt.step()

        preds = torch.argmax(output,dim=1)==(target.cuda(gpu, non_blocking=True))
        
        accuracy.update(preds.cpu())

        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 :
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                    .format(epoch, i, len(loader), batch_time=batch_time,
                            data_time=data_time, loss=losses))
    if args.rank==0:
        wandb.log({"train_loss":losses.avg.item() , "train_accuracy":accuracy.avg},step=epoch)

def eval(epoch,model,loader,crit,args,gpu,stats_file):
    model.eval()
    losses = AverageMeter()
    accuracy = Metric() # ! define this
    with torch.no_grad():
        for step, (input_tensor, targets) in enumerate(loader):
            if torch.cuda.is_available():
                input_tensor =input_tensor.cuda(gpu ,non_blocking=True)
                targets = targets.cuda(gpu,non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(input_tensor)
                loss = crit(outputs, targets)
                preds = torch.argmax(outputs,dim=1)==targets

            accuracy.update(preds.cpu())# ! need to be in cpu for metric to work
            losses.update(loss, input_tensor.size(0))
    
    wandb.log({"test_accuracy": accuracy.avg,"test_loss": losses.avg.item()}, step=epoch)

def main():
    parser=get_downstream_parser()
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    
    # single-node distributed training
    args.rank = 0
    args.dist_url = 'tcp://localhost:58362'
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


if __name__ == '__main__':
    main()
