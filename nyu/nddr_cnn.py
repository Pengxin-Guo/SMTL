import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet

from resnet_dilated import ResnetDilated
from aspp import DeepLabHead
from resnet import Bottleneck, conv1x1


class NDDRLayer(nn.Module):
    def __init__(self, tasks, channels, alpha, beta):
        super(NDDRLayer, self).__init__()
        self.tasks = tasks
        self.layer = nn.ModuleDict({task: nn.Sequential(
                                        nn.Conv2d(len(tasks) * channels, channels, 1, 1, 0, bias=False), nn.BatchNorm2d(channels, momentum=0.05), nn.ReLU()) for task in self.tasks}) # Momentum set as NDDR-CNN repo
        
        # Initialize
        for i, task in enumerate(self.tasks):
            layer = self.layer[task]
            t_alpha = torch.diag(torch.FloatTensor([alpha for _ in range(channels)])) # C x C
            t_beta = torch.diag(torch.FloatTensor([beta for _ in range(channels)])).repeat(1, len(self.tasks)) # C x (C x T)
            t_alpha = t_alpha.view(channels, channels, 1, 1)
            t_beta = t_beta.view(channels, channels * len(self.tasks), 1, 1)
    
            layer[0].weight.data.copy_(t_beta)
            layer[0].weight.data[:,int(i*channels):int((i+1)*channels)].copy_(t_alpha)
            layer[1].weight.data.fill_(1.0)
            layer[1].bias.data.fill_(0.0)


    def forward(self, x):
        x = torch.cat([x[task] for task in self.tasks], 1) # Use self.tasks to retain order!
        output = {task: self.layer[task](x) for task in self.tasks}
        return output


class NDDRCNN(nn.Module):
    def __init__(self, dataset='NYUv2'):
        super(NDDRCNN, self).__init__()

        # ch = [256, 512, 1024, 2048]

        if dataset == 'NYUv2':
            self.class_nb = 13
            self.tasks = ['segmentation', 'depth', 'normal']
            self.num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}
        elif dataset == 'CityScape':
            self.class_nb = 7
            self.tasks = ['segmentation', 'depth']
            self.num_out_channels = {'segmentation': 7, 'depth': 1}
        else:
            raise('No support {} dataset'.format(dataset))
        
        self.backbone = nn.ModuleDict({t: ResnetDilated(resnet.__dict__['resnet50'](pretrained=True)) for t in self.tasks})
        self.heads = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])

        self.all_stages = ['conv', 'layer1_without_conv', 'layer2', 'layer3', 'layer4']
        self.nddr_stages = ['conv', 'layer1_without_conv', 'layer2', 'layer3', 'layer4']
        channels = {'conv': 64, 'layer1_without_conv': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}

        alpha = 0.9
        beta = 0.1

        # NDDR-CNN units
        self.nddr = nn.ModuleDict({stage: NDDRLayer(self.tasks, channels[stage], alpha, beta) for stage in self.nddr_stages})


    def forward(self, x):
        img_size = x.size()[-2:]
        x = {task: x for task in self.tasks} # Feed as input to every single-task network

        # Backbone
        for stage in self.all_stages:
    
            # Forward through next stage of task-specific network
            for task in self.tasks:
                x[task] = self.backbone[task].forward_stage(x[task], stage)
            
            if stage in self.nddr_stages:
                # Fuse task-specific features through NDDR-layer.
                x = self.nddr[stage](x)

        # Task-specific heads
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.heads[i](x[t]), img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out

    def predict(self, x):
        return self.forward(x)
