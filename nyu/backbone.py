import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet

from resnet_dilated import ResnetDilated
from aspp import DeepLabHead
from resnet import Bottleneck, conv1x1


class DeepLabv3(nn.Module):
    def __init__(self, dataset='NYUv2'):
        super(DeepLabv3, self).__init__()
        self.backbone = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        ch = [256, 512, 1024, 2048]

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
        
        self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
        img_size  = x.size()[-2:]
        x = self.backbone(x)
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](x), img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
    
    def predict(self, x):
        return self.forward(x)
        

class Cross_Stitch(nn.Module):
    def __init__(self, dataset='NYUv2'):
        super(Cross_Stitch, self).__init__()
        
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
        
        backbone = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)
        
        backbones = nn.ModuleList([ResnetDilated(resnet.__dict__['resnet50'](pretrained=True)) for _ in self.tasks])
        ch = [256, 512, 1024, 2048]

        # We will apply the cross-stitch unit over the last bottleneck layer in the ResNet. 
        self.resnet_layer1 = nn.ModuleList([])
        self.resnet_layer2 = nn.ModuleList([])
        self.resnet_layer3 = nn.ModuleList([])
        self.resnet_layer4 = nn.ModuleList([])
        for i in range(len(self.tasks)):
            self.resnet_layer1.append(backbones[i].layer1) 
            self.resnet_layer2.append(backbones[i].layer2)
            self.resnet_layer3.append(backbones[i].layer3)
            self.resnet_layer4.append(backbones[i].layer4)

        # define cross-stitch units
        self.cross_unit = nn.Parameter(data=torch.ones(4, 3))
        
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
        img_size  = x.size()[-2:]
        # Shared convolution
        x = self.shared_conv(x)
        
        # ResNet blocks with cross-stitch
        res_feature = [0, 0, 0]
        for j in range(3):
            res_feature[j] = [0, 0, 0, 0]
               
        for i in range(4):
            if i == 0:
                res_layer = self.resnet_layer1
            elif i == 1:
                res_layer = self.resnet_layer2
            elif i == 2:
                res_layer = self.resnet_layer3
            elif i == 3:
                res_layer = self.resnet_layer4
            for j in range(3):
                if i == 0:
                    res_feature[j][i] = res_layer[j](x)
                else:
                    cross_stitch = self.cross_unit[i - 1][0] * res_feature[0][i - 1] + \
                                   self.cross_unit[i - 1][1] * res_feature[1][i - 1] + \
                                   self.cross_unit[i - 1][2] * res_feature[2][i - 1]
                    res_feature[j][i] = res_layer[j](cross_stitch)
            
        # Task specific decoders
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](res_feature[i][-1]), size=img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
        
    def predict(self, x):
        return self.forward(x)

        
class MTANDeepLabv3(nn.Module):
    def __init__(self, dataset='NYUv2'):
        super(MTANDeepLabv3, self).__init__()
        backbone = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        ch = [256, 512, 1024, 2048]
        
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
        
        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)

        # We will apply the attention over the last bottleneck layer in the ResNet. 
        self.shared_layer1_b = backbone.layer1[:-1] 
        self.shared_layer1_t = backbone.layer1[-1]

        self.shared_layer2_b = backbone.layer2[:-1]
        self.shared_layer2_t = backbone.layer2[-1]

        self.shared_layer3_b = backbone.layer3[:-1]
        self.shared_layer3_t = backbone.layer3[-1]

        self.shared_layer4_b = backbone.layer4[:-1]
        self.shared_layer4_t = backbone.layer4[-1]

        # Define task specific attention modules using a similar bottleneck design in residual block
        # (to avoid large computations)
        self.encoder_att_1 = nn.ModuleList([self.att_layer(ch[0], ch[0] // 4, ch[0]) for _ in self.tasks])
        self.encoder_att_2 = nn.ModuleList([self.att_layer(2 * ch[1], ch[1] // 4, ch[1]) for _ in self.tasks])
        self.encoder_att_3 = nn.ModuleList([self.att_layer(2 * ch[2], ch[2] // 4, ch[2]) for _ in self.tasks])
        self.encoder_att_4 = nn.ModuleList([self.att_layer(2 * ch[3], ch[3] // 4, ch[3]) for _ in self.tasks])

        # Define task shared attention encoders using residual bottleneck layers
        # We do not apply shared attention encoders at the last layer,
        # so the attended features will be directly fed into the task-specific decoders.
        self.encoder_block_att_1 = self.conv_layer(ch[0], ch[1] // 4)
        self.encoder_block_att_2 = self.conv_layer(ch[1], ch[2] // 4)
        self.encoder_block_att_3 = self.conv_layer(ch[2], ch[3] // 4)
        
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
        img_size  = x.size()[-2:]
        # Shared convolution
        x = self.shared_conv(x)
        
        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)
        
        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)
        
        # Attention block 1 -> Apply attention over last residual block
        a_1_mask = [att_i(u_1_b) for att_i in self.encoder_att_1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]
        
        # Attention block 2 -> Apply attention over last residual block
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.encoder_block_att_2(a_2_i) for a_2_i in a_2]
        
        # Attention block 3 -> Apply attention over last residual block
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]
        
        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
        a_4 = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]
        
        # Task specific decoders
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](a_4[i]), size=img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
    
    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())
        
    def conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(conv1x1(in_channel, 4 * out_channel, stride=1),
                                   nn.BatchNorm2d(4 * out_channel))
        return Bottleneck(in_channel, out_channel, downsample=downsample)
    
    def predict(self, x):
        return self.forward(x)
        
        
class AdaShare(nn.Module):
    def __init__(self, dataset='NYUv2'):
        super(AdaShare, self).__init__()
        
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
        
        backbone = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)

        # We will apply the task-specific policy over the last bottleneck layer in the ResNet. 
        self.resnet_layer1_d = backbone.layer1[:1]
        self.resnet_layer1_b = backbone.layer1[1:-1]
        self.resnet_layer2_d = backbone.layer2[:1]
        self.resnet_layer2_b = backbone.layer2[1:-1]
        self.resnet_layer3_d = backbone.layer3[:1]
        self.resnet_layer3_b = backbone.layer3[1:-1]
        self.resnet_layer4_d = backbone.layer4[:1]
        self.resnet_layer4_b = backbone.layer4[1:-1]

        # define task-specific policy parameters
        self.alpha = nn.Parameter(torch.FloatTensor(4, len(self.tasks)))
        self.alpha.data.fill_(0)
        
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
        img_size  = x.size()[-2:]
        # Shared convolution
        x = self.shared_conv(x)
        
        # ResNet blocks with task-specific policy
        res_feature = [0, 0, 0]
        for j in range(3):
            res_feature[j] = [0, 0, 0, 0]
               
        for i in range(4):
            if i == 0:
                res_layer_d = self.resnet_layer1_d
                res_layer_b = self.resnet_layer1_b
            elif i == 1:
                res_layer_d = self.resnet_layer2_d
                res_layer_b = self.resnet_layer2_b
            elif i == 2:
                res_layer_d = self.resnet_layer3_d
                res_layer_b = self.resnet_layer3_b
            elif i == 3:
                res_layer_d = self.resnet_layer4_d
                res_layer_b = self.resnet_layer4_b
            for j in range(3):
                # task-specific policy
                temp = torch.sigmoid(self.alpha[i][j])
                temp_alpha = torch.stack([1-temp, temp])
                temp_alpha = F.gumbel_softmax(torch.log(temp_alpha), tau=0.1, hard=True)
                if i == 0:
                    temp_feature = res_layer_d(x)
                else:
                    temp_feature = res_layer_d(res_feature[j][i-1])
                res_feature[j][i] = temp_alpha[0] * temp_feature + temp_alpha[1] * res_layer_b(temp_feature)
            
        # Task specific decoders
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](res_feature[i][-1]), size=img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
        
    def predict(self, x):
        img_size  = x.size()[-2:]
        # Shared convolution
        x = self.shared_conv(x)
        
        # ResNet blocks with task-specific policy
        res_feature = [0, 0, 0]
        for j in range(3):
            res_feature[j] = [0, 0, 0, 0]
               
        for i in range(4):
            if i == 0:
                res_layer_d = self.resnet_layer1_d
                res_layer_b = self.resnet_layer1_b
            elif i == 1:
                res_layer_d = self.resnet_layer2_d
                res_layer_b = self.resnet_layer2_b
            elif i == 2:
                res_layer_d = self.resnet_layer3_d
                res_layer_b = self.resnet_layer3_b
            elif i == 3:
                res_layer_d = self.resnet_layer4_d
                res_layer_b = self.resnet_layer4_b
            for j in range(3):
                # task-specific policy
                temp = torch.sigmoid(self.alpha[i][j])
                if temp >= 0.5:
                    temp_alpha = [0, 1]
                else:
                    temp_alpha = [1, 0]
                if i == 0:
                    temp_feature = res_layer_d(x)
                else:
                    temp_feature = res_layer_d(res_feature[j][i-1])
                res_feature[j][i] = temp_alpha[0] * temp_feature + temp_alpha[1] * res_layer_b(temp_feature)
            
        # Task specific decoders
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](res_feature[i][-1]), size=img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
        
    def get_policy_parameter(self):
        return self.alpha


# do selection at hidden layer
class SMTLmodel(nn.Module):
    def __init__(self, dataset='NYUv2', version='v1'):
        super(SMTLmodel, self).__init__()
        self.version = version
        # shared encoder
        self.backbone_s = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        ch = [256, 512, 1024, 2048]
        
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
        
        # adaptative parameters
        if self.version == 'v1' or self.version =='v2':
            # SMTL-v1 and v2
            self.alpha = nn.Parameter(torch.FloatTensor(len(self.tasks), 2))
            self.alpha.data.fill_(0.5)   # init 0.5(shared) 0.5(specific)
            # self.alpha.data[:,0].fill_(0)  # shared
            # self.alpha.data[:,1].fill_(1)  # specific
        elif self.version == 'v3':
            # SMTL-v3, gumbel softmax
            self.alpha = nn.Parameter(torch.FloatTensor(len(self.tasks)))
            self.alpha.data.fill_(0)
        else:
            print("No correct version parameter!")
            exit()
        
        # task-specific decoder
        self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
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
                temp_alpha = F.softmax(self.alpha[i], 0)     # SMTL-v1, alpha_1 + alpha_2 = 1
            elif self.version == 'v2':
                temp_alpha = torch.exp(self.alpha[i]) / (1 + torch.exp(self.alpha[i])) # SMTL-v2, 0 <= alpha <=1
            elif self.version == 'v3':
                # below for SMTL-v3, gumbel softmax
                temp = torch.sigmoid(self.alpha[i])
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
        
    def predict(self, x):
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
                temp_alpha = F.softmax(self.alpha[i], 0)     # SMTL-v1, alpha_1 + alpha_2 = 1
            elif self.version == 'v2':
                temp_alpha = torch.exp(self.alpha[i]) / (1 + torch.exp(self.alpha[i])) # SMTL-v2, 0 <= alpha <=1
            elif self.version == 'v3':
                # below for SMTL-v3, gumbel softmax
                temp = torch.sigmoid(self.alpha[i])
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
        
    def get_adaptative_parameter(self):
        return self.alpha
        

# do selection at classifier layer
class SMTLmodel_new(nn.Module):
    def __init__(self, dataset='NYUv2', version='v1'):
        super(SMTLmodel_new, self).__init__()
        self.version = version
        # shared encoder
        self.backbone_s = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        ch = [256, 512, 1024, 2048]
        
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
        
        # adaptative parameters
        if self.version == 'v1' or self.version =='v2':
            # SMTL-v1 and v2
            self.alpha = nn.Parameter(torch.FloatTensor(len(self.tasks), 2))
            self.alpha.data.fill_(0.5)   # init 0.5(shared) 0.5(specific)
            # self.alpha.data[:,0].fill_(0)  # shared
            # self.alpha.data[:,1].fill_(1)  # specific
        elif self.version == 'v3':
            # SMTL-v3, gumbel softmax
            self.alpha = nn.Parameter(torch.FloatTensor(len(self.tasks)))
            self.alpha.data.fill_(0)
        else:
            print("No correct version parameter!")
            exit()
        
        # task-specific decoder for shared encoder
        self.decoders_s = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        # task-specific decoder for task-specific encoder
        self.decoders_t = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
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
                temp_alpha = F.softmax(self.alpha[i], 0)     # SMTL-v1, alpha_1 + alpha_2 = 1
            elif self.version == 'v2':
                temp_alpha = torch.exp(self.alpha[i]) / (1 + torch.exp(self.alpha[i])) # SMTL-v2, 0 <= alpha <=1
            elif self.version == 'v3':
                # below for SMTL-v3, gumbel softmax
                temp = torch.sigmoid(self.alpha[i])
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
        
    def predict(self, x):
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
                temp_alpha = F.softmax(self.alpha[i], 0)     # SMTL-v1, alpha_1 + alpha_2 = 1
            elif self.version == 'v2':
                temp_alpha = torch.exp(self.alpha[i]) / (1 + torch.exp(self.alpha[i])) # SMTL-v2, 0 <= alpha <=1
            elif self.version == 'v3':
                # below for SMTL-v3, gumbel softmax
                temp = torch.sigmoid(self.alpha[i])
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
        
    def get_adaptative_parameter(self):
        return self.alpha


# do selection at hidden layer
class SMTLmodel_weight(nn.Module):
    def __init__(self, dataset='NYUv2', version='v1', weighting='EW'):
        super(SMTLmodel_weight, self).__init__()        
        self.version = version
               
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
        
        # for different weighting
        rep_detach = True if weighting in ['GradDrop', 'IMTL', 'MGDA_approx', 'GradNorm'] else False
        if weighting == 'IMTL':
            loss_weight_init = 0.0
        elif weighting == 'UW':
            loss_weight_init = -0.5
        elif weighting == 'WGLS':
            loss_weight_init = 1.0
        elif weighting == 'GradNorm':
            loss_weight_init = 1.0
        else:
            loss_weight_init = None
        m_weighting = 'GMM' if weighting == 'GMM' else None
        
        if rep_detach:
            self.rep = [0] * len(self.tasks)
            self.rep_i = [0] * len(self.tasks)
        if isinstance(loss_weight_init, float):
           self.loss_scale = nn.Parameter(torch.FloatTensor([loss_weight_init] * len(self.tasks)))
        if weighting == 'GMM':
            self.mix_k = 100
            self.comp_mu = nn.Parameter(torch.ones(self.mix_k, len(self.tasks)))
            self.comp_sigma = nn.Parameter(torch.zeros(self.mix_k, len(self.tasks)))
        
        # shared encoder
        self.backbone_s = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        ch = [256, 512, 1024, 2048]
        
        # task-specific encoder
        self.backbone_t = nn.ModuleList([ResnetDilated(resnet.__dict__['resnet50'](pretrained=True)) for _ in self.tasks])
        
        # adaptative parameters
        if self.version == 'v1' or self.version =='v2':
            # SMTL-v1 and v2
            self.alpha = nn.Parameter(torch.FloatTensor(len(self.tasks), 2))
            self.alpha.data.fill_(0.5)   # init 0.5(shared) 0.5(specific)
            # self.alpha.data[:,0].fill_(0)  # shared
            # self.alpha.data[:,1].fill_(1)  # specific
        elif self.version == 'v3':
            # SMTL-v3, gumbel softmax
            self.alpha = nn.Parameter(torch.FloatTensor(len(self.tasks)))
            self.alpha.data.fill_(0)
        else:
            print("No correct version parameter!")
            exit()
        
        # task-specific decoder
        self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
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
                temp_alpha = F.softmax(self.alpha[i], 0)     # SMTL-v1, alpha_1 + alpha_2 = 1
            elif self.version == 'v2':
                temp_alpha = torch.exp(self.alpha[i]) / (1 + torch.exp(self.alpha[i])) # SMTL-v2, 0 <= alpha <=1
            elif self.version == 'v3':
                # below for SMTL-v3, gumbel softmax
                temp = torch.sigmoid(self.alpha[i])
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
        
    def predict(self, x):
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
                temp_alpha = F.softmax(self.alpha[i], 0)     # SMTL-v1, alpha_1 + alpha_2 = 1
            elif self.version == 'v2':
                temp_alpha = torch.exp(self.alpha[i]) / (1 + torch.exp(self.alpha[i])) # SMTL-v2, 0 <= alpha <=1
            elif self.version == 'v3':
                # below for SMTL-v3, gumbel softmax
                temp = torch.sigmoid(self.alpha[i])
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
    
    def GradDrop_backward(self, losses, leak=0.0):
        epsilon = 1e-7
        if isinstance(self.rep, list):
            batch_size = self.rep[0].size()[0]
            rep_size = self.rep[0].size()
            inputs = torch.stack(self.rep).reshape(len(self.tasks), batch_size, -1)
        else:
            batch_size = self.rep.size()[0]
            rep_size = self.rep.size()
            inputs = self.rep.reshape(batch_size, -1).unsqueeze(0).repeat(len(self.tasks), 1, 1)
        per_loss_grads = torch.zeros(len(self.tasks), *rep_size).cuda()
        for tn, loss in enumerate(losses):
            loss.backward(retain_graph=True)
            per_loss_grads[tn] = self.rep_i[tn].grad.data.clone()
        grads = per_loss_grads.reshape(len(self.tasks), batch_size, -1)
        inputs_abs = torch.abs((torch.abs(inputs)<=epsilon).float()+inputs)
        grads = grads*(inputs/inputs_abs)
        grads = grads.sum(1)

        grad_sign_positive = (grads>0.0).float()
        grad_sign_negative = (grads<0.0).float()

        prob_pos = grads.sum(0)/(2*grads.abs().sum(0)+epsilon)+0.5
        prob_pos = (prob_pos>=torch.rand(prob_pos.shape).cuda()).float()-0.5

        grad_masks = (((grad_sign_positive-grad_sign_negative)*prob_pos)>=0).float()
        grad_masks = grad_masks.unsqueeze(1).repeat(1, batch_size, 1).reshape(*per_loss_grads.size())
        
        transformed_per_loss_grads = per_loss_grads*(leak+(1-leak)*grad_masks)
        
        if isinstance(self.rep, list):
            for tn in range(len(self.tasks)):
                self.rep[tn].backward(transformed_per_loss_grads[tn], retain_graph=True)
        else:
            self.rep.backward(transformed_per_loss_grads.sum(0))
    
        # transformed_grad_norm = (transformed_grad**2).sum().sqrt()
        # original_grad_norm = (grad_output**2).sum().sqrt()

        # grad_input = transformed_grad*original_grad_norm/(transformed_grad_norm+epsilon)
#         self.rep.backward(transformed_grad)
        
        self.rep_i = [0]*len(self.tasks)
    
    def IMTL_backward(self, losses):
        if isinstance(self.rep, list):
            batch_size = self.rep[0].size()[0]
            rep_size = self.rep[0].size()
        else:
            batch_size = self.rep.size()[0]
            rep_size = self.rep.size()
        per_loss_grads = torch.zeros(len(self.tasks), *rep_size).cuda()
        for tn, loss in enumerate(losses):
            loss = torch.exp(self.loss_scale[tn])*loss - self.loss_scale[tn]
            loss.backward(retain_graph=True)
            per_loss_grads[tn] = self.rep_i[tn].grad.data.clone()
        grads = per_loss_grads.reshape(len(self.tasks), batch_size, -1).sum(1)
        grads_unit = grads/(torch.norm(grads, p=2, dim=-1, keepdim=True)+1e-8)

        D = grads[0:1].repeat(len(self.tasks)-1, 1) - grads[1:]
        U = grads_unit[0:1].repeat(len(self.tasks)-1, 1) - grads_unit[1:]

        alpha = torch.matmul(torch.matmul(grads[0], U.t()), torch.linalg.inv(torch.matmul(D, U.t())))
        alpha = torch.cat((1-alpha.sum().unsqueeze(0), alpha), dim=0)
        
        if isinstance(self.rep, list):
            for tn in range(len(self.tasks)):
                self.rep[tn].backward(alpha[tn]*per_loss_grads[tn], retain_graph=True)
        else:
            transformed_grad = sum(alpha[tn]*per_loss_grads[tn] for tn in range(len(self.tasks)))
            self.rep.backward(transformed_grad)
        self.rep_i = [0]*len(self.tasks)
        return alpha

    def MGDA_approx_backward(self, losses, mgda_gn='none'):
        if isinstance(self.rep, list):
            batch_size = self.rep[0].size()[0]
            rep_size = self.rep[0].size()
        else:
            batch_size = self.rep.size()[0]
            rep_size = self.rep.size()
        per_loss_grads = torch.zeros(len(self.tasks), *rep_size).cuda()
        for tn, loss in enumerate(losses):
            loss.backward(retain_graph=True)
            per_loss_grads[tn] = self.rep_i[tn].grad.data.clone()

        grads = {tn:per_loss_grads[tn] for tn in range(len(self.tasks))}
        loss_data = {tn:losses[tn].item() for tn in range(len(self.tasks))}
        gn = gradient_normalizers(grads, loss_data, normalization_type=mgda_gn)
        for tn in range(len(self.tasks)):
            grads[tn] = grads[tn] / gn[tn]
        sol, _ = MinNormSolver.find_min_norm_element([grads[tn] for tn in range(len(self.tasks))])
        if isinstance(self.rep, list):
            for tn in range(len(self.tasks)):
                self.rep[tn].backward(float(sol[tn])*per_loss_grads[tn], retain_graph=True)
        else:
            transformed_grad = sum(float(sol[tn])*per_loss_grads[tn] for tn in range(len(self.tasks)))
            self.rep.backward(transformed_grad)
        self.rep_i = [0]*len(self.tasks)
        return torch.Tensor(sol).cuda()
    
    def GradNorm_backward(self, losses, L_0, alpha=1.5):
        loss_scale = len(self.tasks)/self.loss_scale.sum()*self.loss_scale
        if isinstance(self.rep, list):
            batch_size = self.rep[0].size()[0]
            rep_size = self.rep[0].size()
        else:
            batch_size = self.rep.size()[0]
            rep_size = self.rep.size()
        per_loss_grads = torch.zeros(len(self.tasks), *rep_size).cuda()
        for tn, loss in enumerate(losses):
            loss.backward(retain_graph=True)
            per_loss_grads[tn] = self.rep_i[tn].grad.data.clone()
        grads = per_loss_grads.reshape(len(self.tasks), batch_size, -1).sum(1).detach()
        G_per_loss = torch.norm(loss_scale.unsqueeze(1)*grads, p=2, dim=-1)
        G = G_per_loss.mean(0)
        L_i = torch.Tensor([losses[tn].item()/L_0[tn] for tn in range(len(self.tasks))])
        r_i = L_i/L_i.mean()
        constant_term = (G*(r_i[tn]**alpha)).detach()
        L_grad = sum([torch.abs(G_per_loss[tn]-constant_term) for tn in range(len(self.tasks))])
        L_grad.backward()
        loss_weight = loss_scale.detach().clone()
        if isinstance(self.rep, list):
            for tn in range(len(self.tasks)):
                self.rep[tn].backward(loss_weight[tn]*per_loss_grads[tn], retain_graph=True)
        else:
            transformed_grad = sum(loss_weight[tn]*per_loss_grads[tn] for tn in range(len(self.tasks)))
            self.rep.backward(transformed_grad)
        
    def get_adaptative_parameter(self):
        return self.alpha