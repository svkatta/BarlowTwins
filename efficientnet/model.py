
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn


class DownstreamClassifer(nn.Module):
    def __init__(self, no_of_classes =256,final_pooling_type="Avg"):
        super(DownstreamClassifer, self).__init__()
        self.backbone = EfficientNet.from_name('efficientnet-b0',
                                                    final_pooling_type=final_pooling_type,
                                                    include_top = False,
                                                    in_channels = 1,
                                                    image_size = None)
        self.classifier = nn.Linear(1280,no_of_classes)

    def forward(self,batch):
        x = self.backbone(batch)
        x = x.flatten(start_dim=1) #1280 (already swished)
        x = self.classifier(x)        
        return x



class BarlowTwins(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = EfficientNet.from_name('efficientnet-b0',
                                                    final_pooling_type=args.final_pooling_type,
                                                    include_top = False,
                                                    in_channels = 1,
                                                    image_size = None) # (,)==> (1280)
        

        # projector
        self.projector = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(1280, 8192, bias=False),
                                        nn.BatchNorm1d(8192),
                                        nn.ReLU(),
                                        nn.Linear(8192, 8192, bias = False))
        
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(8192, affine=False)

    def forward(self, y1, y2):
        z1 = self.backbone(y1)
        z2 = self.backbone(y2) # BS , 1280 ,1 ,1

        z1_flat=z1.flatten(start_dim=1)   # BS x ddim
        z2_flat=z2.flatten(start_dim=1)   # BS x ddim
        
        z1_normalised = self.bn(self.projector(z1_flat))
        z2_normalised = self.bn(self.projector(z2_flat))
        # empirical cross-correlation matrix
        c = z1_normalised.T @ z2_normalised

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss , on_diag , off_diag
    
    def off_diagonal(self,x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTriplets(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = EfficientNet.from_name('efficientnet-b0',
                                                    final_pooling_type=args.final_pooling_type,
                                                    include_top = False,
                                                    in_channels = 1,
                                                    image_size = None) # (,)==> (1280)
        

        # projector
        self.projector = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(1280, 8192, bias=False),
                                        nn.BatchNorm1d(8192),
                                        nn.ReLU(),
                                        nn.Linear(8192, 8192, bias = False))
        
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(8192, affine=False)

    def forward(self, y1, y2,y3):
        z1 = self.backbone(y1) #(BS,1280,1,1)
        z2 = self.backbone(y2) #(BS,1280,1,1)
        z3 = self.backbone(y3) #(BS,1280,1,1)

        z1_flat=z1.flatten(start_dim=1)   # BS x ddim
        z2_flat=z2.flatten(start_dim=1)   # BS x ddim
        z3_flat=z3.flatten(start_dim=1)

        z1_normalised = self.bn(self.projector(z1_flat))  # BS x ddim
        z2_normalised = self.bn(self.projector(z2_flat))  # BS x ddim
        z3_normalised = self.bn(self.projector(z3_flat))  # BS x ddim

        z1_final = torch.unsqueeze(z1_normalised,dim=0) # BS x 1 x ddim
        z2_final = torch.unsqueeze(z2_normalised,dim=1) # 1 x BS x ddim
        z3_final = torch.unsqueeze(z3_normalised,dim=1) # 1 x BS x ddim

        x,y  = torch.broadcast_tensor(z1_final,z2_final) #  
        cross = torch.cross(x,y,dim=2) # BS x BS x ddim 

        c =  torch.abs(torch.sum(cross * z3_final,dim=-1))  # BS x BS
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).pow_(2).sum()
        off_diag = self.off_diagonal(c).add_(-1).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss , on_diag , off_diag

    def off_diagonal(self,x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


#---------------------------------------------------------------------------------------#


# def __init__(self,args):
    #     super(AAAI_BARLOW, self).__init__()

    #     self.args = args
    #     self.model_efficient = EfficientNet.from_name('efficientnet-b0',include_top = False, in_channels = 1,image_size = None)
    #     self.projector = nn.Sequential(nn.Dropout(0.5),nn.Linear(1280, 8192, bias=False),nn.BatchNorm1d(8192),nn.ReLU(),nn.Linear(8192, 8192, bias = False))
    #     self.bn = nn.BatchNorm1d(8192, affine=False)