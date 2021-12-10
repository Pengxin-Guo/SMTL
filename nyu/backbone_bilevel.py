import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet

from resnet_dilated import ResnetDilated
from aspp import DeepLabHead
from resnet import Bottleneck, conv1x1


# do selection at hidden layer
class SMTLmodel(nn.Module):
    def __init__(self, dataset='NYUv2', version='v1'):
        super(SMTLmodel, self).__init__()
        self.version = version
        # shared encoder
        self.backbone_s = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        
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
        
        # task-specific encoder
        self.backbone_t = nn.ModuleList([ResnetDilated(resnet.__dict__['resnet50'](pretrained=True)) for _ in self.tasks])
        
        # task-specific decoder
        self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x, h):
        alpha = h.get_adaptative_parameter()
        img_size  = x.size()[-2:]
        # shared encoder output
        x_s = self.backbone_s(x)
        # task-specific encoder output
        x_t = [0 for _ in self.tasks]
        for i in range(len(self.tasks)):
            x_t[i] = self.backbone_t[i](x)
        # combine shared encoder output and task-specific encoder output, obtain final hidden feature
        x_h = [0 for _ in self.tasks]
        for i in range(len(self.tasks)):
            if self.version == 'v1':
                temp_alpha = F.softmax(alpha[i], 0)     # SMTL-v1, alpha_1 + alpha_2 = 1
            elif self.version == 'v2':
                temp_alpha = torch.exp(alpha[i]) / (1 + torch.exp(alpha[i])) # SMTL-v2, 0 <= alpha <=1
            elif self.version == 'v3':
                # below for SMTL-v3, gumbel softmax
                temp = torch.sigmoid(alpha[i])
                temp_alpha = torch.stack([1-temp, temp])
                temp_alpha = F.gumbel_softmax(torch.log(temp_alpha), tau=0.1, hard=True)
            else:
                print("No correct version parameter!")
                exit()

            x_h[i] = temp_alpha[0] * x_s + temp_alpha[1] * x_t[i]
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](x_h[i]), img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
        
    def predict(self, x, h):
        alpha = h.get_adaptative_parameter()
        img_size  = x.size()[-2:]
        # shared encoder output
        x_s = self.backbone_s(x)
        # task-specific encoder output
        x_t = [0 for _ in self.tasks]
        for i in range(len(self.tasks)):
            x_t[i] = self.backbone_t[i](x)
        # combine shared encoder output and task-specific encoder output, obtain final hidden feature
        x_h = [0 for _ in self.tasks]
        for i in range(len(self.tasks)):
            if self.version == 'v1':
                temp_alpha = F.softmax(alpha[i], 0)     # SMTL-v1, alpha_1 + alpha_2 = 1
            elif self.version == 'v2':
                temp_alpha = torch.exp(alpha[i]) / (1 + torch.exp(alpha[i])) # SMTL-v2, 0 <= alpha <=1
            elif self.version == 'v3':
                # below for SMTL-v3, gumbel softmax
                temp = torch.sigmoid(alpha[i])
                if temp >= 0.5:
                    temp_alpha = [0, 1]
                else:
                    temp_alpha = [1, 0]
            else:
                print("No correct version parameter!")
                exit()

            x_h[i] = temp_alpha[0] * x_s + temp_alpha[1] * x_t[i]
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](x_h[i]), img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
    
    def get_share_params(self):
        return self.backbone.parameters()
        

# do selection at classifier layer
class SMTLmodel_new(nn.Module):
    def __init__(self, dataset='NYUv2', version='v1'):
        super(SMTLmodel_new, self).__init__()
        self.version = version
        # shared encoder
        self.backbone_s = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        
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
        
        # task-specific encoder
        self.backbone_t = nn.ModuleList([ResnetDilated(resnet.__dict__['resnet50'](pretrained=True)) for _ in self.tasks])
        
        # task-specific decoder for shared encoder
        self.decoders_s = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        # task-specific decoder for task-specific encoder
        self.decoders_t = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x, h):
        alpha = h.get_adaptative_parameter()
        img_size  = x.size()[-2:]
        # shared encoder output
        x_s = self.backbone_s(x)
        # task-specific encoder output
        x_t = [0 for _ in self.tasks]
        for i in range(len(self.tasks)):
            x_t[i] = self.backbone_t[i](x)
        # shared decoder output
        out_s = [0 for _ in self.tasks]
        # task-specific decoder output
        out_t = [0 for _ in self.tasks]
        # combine shared decoder output and task-specific decoder output, obtain final output
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out_s[i] = F.interpolate(self.decoders_s[i](x_s), img_size, mode='bilinear', align_corners=True)
            out_t[i] = F.interpolate(self.decoders_t[i](x_t[i]), img_size, mode='bilinear', align_corners=True)
            if self.version == 'v1':
                temp_alpha = F.softmax(alpha[i], 0)     # SMTL-v1, alpha_1 + alpha_2 = 1
            elif self.version == 'v2':
                temp_alpha = torch.exp(alpha[i]) / (1 + torch.exp(alpha[i])) # SMTL-v2, 0 <= alpha <=1
            elif self.version == 'v3':
                # below for SMTL-v3, gumbel softmax
                temp = torch.sigmoid(alpha[i])
                temp_alpha = torch.stack([1-temp, temp])
                temp_alpha = F.gumbel_softmax(torch.log(temp_alpha), tau=0.1, hard=True)
            else:
                print("No correct version parameter!")
                exit()
            out[i] = temp_alpha[0] * out_s[i] + temp_alpha[1] * out_t[i]    
            
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        
        return out
        
    def predict(self, x, h):
        alpha = h.get_adaptative_parameter()
        img_size  = x.size()[-2:]
        # shared encoder output
        x_s = self.backbone_s(x)
        # task-specific encoder output
        x_t = [0 for _ in self.tasks]
        for i in range(len(self.tasks)):
            x_t[i] = self.backbone_t[i](x)
        # shared decoder output
        out_s = [0 for _ in self.tasks]
        # task-specific decoder output
        out_t = [0 for _ in self.tasks]
        # combine shared decoder output and task-specific decoder output, obtain final output
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out_s[i] = F.interpolate(self.decoders_s[i](x_s), img_size, mode='bilinear', align_corners=True)
            out_t[i] = F.interpolate(self.decoders_t[i](x_t[i]), img_size, mode='bilinear', align_corners=True)
            if self.version == 'v1':
                temp_alpha = F.softmax(alpha[i], 0)     # SMTL-v1, alpha_1 + alpha_2 = 1
            elif self.version == 'v2':
                temp_alpha = torch.exp(alpha[i]) / (1 + torch.exp(alpha[i])) # SMTL-v2, 0 <= alpha <=1
            elif self.version == 'v3':
                # below for SMTL-v3, gumbel softmax
                temp = torch.sigmoid(alpha[i])
                if temp >= 0.5:
                    temp_alpha = [0, 1]
                else:
                    temp_alpha = [1, 0]
            else:
                print("No correct version parameter!")
                exit()
            out[i] = temp_alpha[0] * out_s[i] + temp_alpha[1] * out_t[i]    
            
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        
        return out
    
    def get_share_params(self):
        return self.backbone.parameters()
        