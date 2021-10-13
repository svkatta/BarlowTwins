
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn

class DeepCluster_ICASSP(nn.Module):
    def __init__(self, no_of_classes =256,final_pooling_type="Avg"):
        super(DeepCluster_ICASSP, self).__init__()
        self.model_efficient = EfficientNet.from_name('efficientnet-b0',
                                                    final_pooling_type=final_pooling_type,
                                                    include_top = False,
                                                    in_channels = 1,
                                                    image_size = None)
        self.classifier = nn.Sequential(
                            nn.Dropout(0.5),nn.Linear(1280, 512),nn.ReLU(),
                            nn.Dropout(0.5),nn.Linear(512, 512),nn.ReLU())
        self.top_layer = nn.Linear(512,no_of_classes)

    def forward(self,batch):
        x = self.model_efficient(batch)
        x = x.flatten(start_dim=1) #1280 (already swished)
        x = self.classifier(x)
        
        if self.top_layer:
            x = self.top_layer(x)
        
        return x

class DeepCluster_downstream(nn.Module):
    def __init__(self, no_of_classes =256,final_pooling_type="Avg"):
        super(DeepCluster_downstream, self).__init__()
        self.model_efficient = EfficientNet.from_name('efficientnet-b0',
                                                    final_pooling_type=final_pooling_type,
                                                    include_top = False,
                                                    in_channels = 1,
                                                    image_size = None)
        self.classifier = nn.Linear(1280,no_of_classes)

    def forward(self,batch):
        x = self.model_efficient(batch)
        x = x.flatten(start_dim=1) #1280 (already swished)
        x = self.classifier(x)        
        return x



class BarlowTwins(nn.Module):
    # def __init__(self,args):
    #     super(AAAI_BARLOW, self).__init__()

    #     self.args = args
    #     self.model_efficient = EfficientNet.from_name('efficientnet-b0',include_top = False, in_channels = 1,image_size = None)
    #     self.projector = nn.Sequential(nn.Dropout(0.5),nn.Linear(1280, 8192, bias=False),nn.BatchNorm1d(8192),nn.ReLU(),nn.Linear(8192, 8192, bias = False))
    #     self.bn = nn.BatchNorm1d(8192, affine=False)


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
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        z1_flat=z1.flatten(start_dim=1)   # BS x ddim
        z2_flat=z2.flatten(start_dim=1)   # BS x ddim

        z1_normalised = self.bn(z1_flat)
        z2_normalised = self.bn(z2_flat)

        # empirical cross-correlation matrix
        c = z1_normalised.T @ z2_normalised

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss
    
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

