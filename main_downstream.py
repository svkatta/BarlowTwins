import logging
import os
import time
import json
import matplotlib.pyplot as plt
import torch
from torch import nn
import sys
from datasets.data_utils import collate_fn_padd_downstream
from datasets.dataset import get_dataset
from collections import OrderedDict
from efficientnet.model import  DownstreamClassifer
from utils import (AverageMeter,Metric,freeze_effnet,get_downstream_parser,load_pretrain)#resume_from_checkpoint, save_to_checkpoint,set_seed
import socketserver

import wandb
# wandb.login()
os.environ["WANDB_API_KEY"] = "52cfe23f2dcf3b889f99716f771f81c71fd75320"
os.environ["WANDB_MODE"] = "offline"
wandb.define_metric("test_loss", summary="min")
wandb.define_metric("test_accuracy", summary="max")
wandb.define_metric("train_loss", summary="min")
wandb.define_metric("train_accuracy", summary="max")

def main_worker(gpu, args):
    args.rank += gpu
    args.ngpus = os.environ["CUDA_VISIBLE_DEVICES"]
    if args.rank==0:
        if args.freeze_effnet :
            extra_name = "freeze"
        else:
            extra_name = "finetune"
        run = wandb.init(project="downstream2",config=vars(args),
            name="_".join([args.backbone,args.final_pooling_type,args.tag,extra_name]))
    print("wandb init done")
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    print("process inited done")
    stats_file=None
    args.exp_root = args.exp_dir / args.tag
    args.exp_root.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True # ! change it set seed 
    print("process inited done")
    # train and test loaders 
    # ! user sampler and ddp 
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    train_dataset,test_dataset = get_dataset(args.down_stream_task)
    print("model done1")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)    
    print("model done2")
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=per_device_batch_size,
                                                collate_fn = collate_fn_padd_downstream,
                                                pin_memory=True,sampler = train_sampler)
    print("model done3")
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=per_device_batch_size,
                                                collate_fn = collate_fn_padd_downstream,
                                                pin_memory=True)  
    print("model done4")
    # models
    args.no_of_classes= train_dataset.no_of_classes
    model = DownstreamClassifer(args)
    print("model done5")
    # Resume
    start_epoch =0 
    if args.resume:
        raise NotImplementedError
        # resume_from_checkpoint(args.pretrain_path,model,optimizer)
    elif args.pretrain_path:
        print("here",args.pretrain_path,args.load_only_efficientNet)
        # load_pretrain(args.pretrain_path,model,args.load_only_efficientNet)
        checkpoint = torch.load(args.pretrain_path,map_location=torch.device('cpu'))
        print("here")
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        for key in new_state_dict.copy():
            if not key.startswith('backbone'):
                del new_state_dict[key]
        mod_missing_keys,mod_unexpected_keys   = model.load_state_dict(new_state_dict,strict=False)
        assert mod_missing_keys == ['classifier.weight', 'classifier.bias'] and mod_unexpected_keys == []
    
    print("laodin done")
    model = model.cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    criterion = nn.CrossEntropyLoss().cuda(gpu)
# -------------------------------optmiser------------- 
    classifier_parameters, model_parameters = [], []
    for name, param in model.named_parameters():
        if name in {'classifier.weight', 'classifier.bias'}:
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)
    param_groups = [dict(params=classifier_parameters, lr=args.lr_classifier)]
    if not args.freeze_effnet:
        print("freezing backbone weights")
        param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
    
    optimizer = torch.optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # optimizer = torch.optim.Adam(
    #     filter(lambda x: x.requires_grad, model.parameters()),
    #     lr=args.lr,
    # )
    print("syncbn done")
# -----------------------------------------
    if args.rank == 0:
        print("Starting To Train")
        wandb.watch(model,criterion=criterion, log="all", log_freq=10)
    print("want done")

    for epoch in range(start_epoch,args.epochs):
        train_sampler.set_epoch(epoch)
        train_one_epoch(train_loader, model, criterion, optimizer, epoch,gpu,args)
        # save_to_checkpoint(args.down_stream_task,args.exp_root,epoch,model,optimizer)

        if args.rank == 0 :
            eval(epoch,model,test_loader,criterion,args,gpu,stats_file)

        scheduler.step() # ! per epoch scheduler step 

    if args.rank==0:
        run.finish()


def train_one_epoch(loader, model, crit, opt, epoch,gpu,args):
    '''
    Train one Epoch
    '''
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
    with socketserver.TCPServer(("localhost", 0), None) as s:
        free_port = s.server_address[1]
    args.rank = 0
    args.dist_url = 'tcp://localhost:'+str(free_port)
    print(args.dist_url)
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


if __name__ == '__main__':
    main()
