import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet


class DMTL(nn.Module):
    def __init__(self, task_num, base_net='resnet50', hidden_dim=1024, class_num=31):
        super(DMTL, self).__init__()
        # base network
        self.base_network = resnet.__dict__[base_net](pretrained=True)
        # shared layer
        self.avgpool = self.base_network.avgpool
        self.hidden_layer_list = [nn.Linear(2048, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.hidden_layer = nn.Sequential(*self.hidden_layer_list)
        # task-specific layer
        self.classifier_parameter = nn.Parameter(torch.FloatTensor(task_num, hidden_dim, class_num))

        # initialization
        self.hidden_layer[0].weight.data.normal_(0, 0.005)
        self.hidden_layer[0].bias.data.fill_(0.1)
        self.classifier_parameter.data.normal_(0, 0.01)

    def forward(self, inputs, task_index):
        features = self.base_network(inputs)
        features = torch.flatten(self.avgpool(features), 1)
        hidden_features = self.hidden_layer(features)
        outputs = torch.mm(hidden_features, self.classifier_parameter[task_index])
        return outputs

    def predict(self, inputs, task_index):
        return self.forward(inputs, task_index)
        

class MTAN_ResNet(nn.Module):
    def __init__(self, task_num, num_classes):
        super(MTAN_ResNet, self).__init__()
        backbone = resnet.__dict__['resnet50'](pretrained=True)
        self.task_num = task_num
        # filter = [64, 128, 256, 512]   # for resent18
        filter = [256, 512, 1024, 2048]

        self.conv1, self.bn1, self.relu1, self.maxpool = backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.linear = nn.ModuleList([nn.Linear(filter[-1], num_classes) for _ in range(self.task_num)])

        # attention modules
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])

        for j in range(self.task_num):
            if j < self.task_num-1:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))

            for i in range(3):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))

        for i in range(3):
            if i < 2:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

    def forward(self, x, k):
        g_encoder = [0] * 4

        atten_encoder = [0] * 4
        for i in range(4):
            atten_encoder[i] = [0] * 3

        # shared encoder
        x = self.maxpool(self.relu1(self.bn1(self.conv1(x))))
        g_encoder[0] = self.layer1(x)
        g_encoder[1] = self.layer2(g_encoder[0])
        g_encoder[2] = self.layer3(g_encoder[1])
        g_encoder[3] = self.layer4(g_encoder[2])

        # apply attention modules
        for j in range(4):
            if j == 0:
                atten_encoder[j][0] = self.encoder_att[k][j](g_encoder[0])
                atten_encoder[j][1] = (atten_encoder[j][0]) * g_encoder[0]
                atten_encoder[j][2] = self.encoder_block_att[j](atten_encoder[j][1])
                atten_encoder[j][2] = F.max_pool2d(atten_encoder[j][2], kernel_size=2, stride=2)
            else:
                atten_encoder[j][0] = self.encoder_att[k][j](torch.cat((g_encoder[j], atten_encoder[j - 1][2]), dim=1))
                atten_encoder[j][1] = (atten_encoder[j][0]) * g_encoder[j]
                atten_encoder[j][2] = self.encoder_block_att[j](atten_encoder[j][1])
                if j < 3:
                    atten_encoder[j][2] = F.max_pool2d(atten_encoder[j][2], kernel_size=2, stride=2)

        pred = self.avgpool(atten_encoder[-1][-1])
        pred = pred.view(pred.size(0), -1)

        out = self.linear[k](pred)
        return out
        
    def predict(self, x, k):
        return self.forward(x, k)
    
    def conv_layer(self, channel):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channel[1]),
            nn.ReLU(inplace=True),
        )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block
        

class AdaShare(nn.Module):
    def __init__(self, task_num, base_net='resnet50', hidden_dim=1024, class_num=31):
        super(AdaShare, self).__init__()
        # base network
        self.base_network = resnet.__dict__[base_net](pretrained=True)
        self.task_num = task_num
        self.shared_conv = nn.Sequential(self.base_network.conv1, self.base_network.bn1, nn.ReLU(inplace=True), self.base_network.maxpool)
        # We will apply the task-specific policy over the last bottleneck layer in the ResNet. 
        self.resnet_layer1_d = self.base_network.layer1[:1]
        self.resnet_layer1_b = self.base_network.layer1[1:-1]
        self.resnet_layer2_d = self.base_network.layer2[:1]
        self.resnet_layer2_b = self.base_network.layer2[1:-1]
        self.resnet_layer3_d = self.base_network.layer3[:1]
        self.resnet_layer3_b = self.base_network.layer3[1:-1]
        self.resnet_layer4_d = self.base_network.layer4[:1]
        self.resnet_layer4_b = self.base_network.layer4[1:-1]
        
        # define task-specific policy parameters
        self.alpha = nn.Parameter(torch.FloatTensor(4, self.task_num))
        self.alpha.data.fill_(0)
        
        # shared layer
        self.avgpool = self.base_network.avgpool
        self.hidden_layer_list = [nn.Linear(2048, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.hidden_layer = nn.Sequential(*self.hidden_layer_list)
        # task-specific layer
        self.classifier_parameter = nn.Parameter(torch.FloatTensor(self.task_num, hidden_dim, class_num))

        # initialization
        self.hidden_layer[0].weight.data.normal_(0, 0.005)
        self.hidden_layer[0].bias.data.fill_(0.1)
        self.classifier_parameter.data.normal_(0, 0.01)

    def forward(self, inputs, task_index):
        # Shared convolution
        inputs = self.shared_conv(inputs)
        # ResNet blocks with task-specific policy
        res_feature = [0, 0, 0, 0]
               
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
    
            temp = torch.sigmoid(self.alpha[i][task_index])
            temp_alpha = torch.stack([1-temp, temp])
            temp_alpha = F.gumbel_softmax(torch.log(temp_alpha), tau=0.1, hard=True)
            if i == 0:
                temp_feature = res_layer_d(inputs)
            else:
                temp_feature = res_layer_d(res_feature[i-1])
            res_feature[i] = temp_alpha[0] * temp_feature + temp_alpha[1] * res_layer_b(temp_feature)
                
        # features = self.base_network(inputs)
        features = torch.flatten(self.avgpool(res_feature[-1]), 1)
        hidden_features = self.hidden_layer(features)
        outputs = torch.mm(hidden_features, self.classifier_parameter[task_index])
        return outputs

    def predict(self, inputs, task_index):
        r# Shared convolution
        inputs = self.shared_conv(inputs)
        # ResNet blocks with task-specific policy
        res_feature = [0, 0, 0, 0]
               
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
    
            temp = torch.sigmoid(self.alpha[i][task_index])
            if temp >= 0.5:
                temp_alpha = [0, 1]
            else:
                temp_alpha = [1, 0]
            if i == 0:
                temp_feature = res_layer_d(inputs)
            else:
                temp_feature = res_layer_d(res_feature[i-1])
            res_feature[i] = temp_alpha[0] * temp_feature + temp_alpha[1] * res_layer_b(temp_feature)
                
        # features = self.base_network(inputs)
        features = torch.flatten(self.avgpool(res_feature[-1]), 1)
        hidden_features = self.hidden_layer(features)
        outputs = torch.mm(hidden_features, self.classifier_parameter[task_index])
        return outputs
        

class AMTL(nn.Module):
    def __init__(self, task_num, base_net='resnet50', hidden_dim=1024, class_num=31, version='v1'):
        super(AMTL, self).__init__()
        # shared base network
        self.base_network_s = resnet.__dict__[base_net](pretrained=True)
        # task-specific base network
        self.base_network_t = nn.ModuleList([resnet.__dict__[base_net](pretrained=True) for _ in range(task_num)])
        self.avgpool = self.base_network_s.avgpool
        # shared hidden layer
        self.hidden_layer_list_s = [nn.Linear(2048, hidden_dim),nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.hidden_layer_s = nn.Sequential(*self.hidden_layer_list_s)
        # task-specific hidden layer
        self.hidden_layer_list_t = [nn.ModuleList([nn.Linear(2048, hidden_dim),nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]) for _ in range(task_num)]
        self.hidden_layer_t = nn.ModuleList([nn.Sequential(*self.hidden_layer_list_t[t]) for t in range(task_num)])
        # classifier layer
        self.classifier_parameter = nn.Parameter(torch.FloatTensor(task_num, hidden_dim, class_num))

        # initialization
        self.hidden_layer_s[0].weight.data.normal_(0, 0.005)
        self.hidden_layer_s[0].bias.data.fill_(0.1)
        self.classifier_parameter.data.normal_(0, 0.01)
        for t in range(task_num):
            self.hidden_layer_t[t][0].weight.data.normal_(0, 0.005)
            self.hidden_layer_t[t][0].bias.data.fill_(0.1)
        
        self.version = version
        
        # adaptative parameters
        if self.version == 'v1' or self.version =='v2':
            # AMTL-v1 and v2
            self.alpha = nn.Parameter(torch.FloatTensor(task_num, 2))
            self.alpha.data.fill_(0.5)   # init 0.5(shared) 0.5(specific)
            # self.alpha.data[:,0].fill_(0)  # shared
            # self.alpha.data[:,1].fill_(1)  # specific
        elif self.version == 'v3':
            # AMTL-v3, gumbel softmax
            self.alpha = nn.Parameter(torch.FloatTensor(task_num))
            self.alpha.data.fill_(0)
        else:
            print("No correct version parameter!")
            exit()

    def forward(self, inputs, task_index):
        features_s = self.base_network_s(inputs)
        features_s = torch.flatten(self.avgpool(features_s), 1)
        hidden_features_s = self.hidden_layer_s(features_s)
        
        features_t = self.base_network_t[task_index](inputs)
        features_t = torch.flatten(self.avgpool(features_t), 1)
        hidden_features_t = self.hidden_layer_t[task_index](features_t)
        
        if self.version == 'v1':
            temp_alpha = F.softmax(self.alpha[task_index], 0)     # AMTL-v1,  alpha_1 + alpha_2 = 1
        elif self.version == 'v2':
            temp_alpha = torch.exp(self.alpha[task_index]) / (1 + torch.exp(self.alpha[task_index])) # AMTL-v2,  0 <= alpha <=1
        elif self.version == 'v3':
            # below for AMTL-v3, gumbel softmax
            temp = torch.sigmoid(self.alpha[task_index])
            temp_alpha = torch.stack([1-temp, temp])
            temp_alpha = F.gumbel_softmax(torch.log(temp_alpha), tau=0.1, hard=True)
        else:
            print("No correct version parameter!")
            exit()

        hidden_features = temp_alpha[0] * hidden_features_s + temp_alpha[1] * hidden_features_t
        
        outputs = torch.mm(hidden_features, self.classifier_parameter[task_index])
        return outputs
    
    def predict(self, inputs, task_index):
        features_s = self.base_network_s(inputs)
        features_s = torch.flatten(self.avgpool(features_s), 1)
        hidden_features_s = self.hidden_layer_s(features_s)
        
        features_t = self.base_network_t[task_index](inputs)
        features_t = torch.flatten(self.avgpool(features_t), 1)
        hidden_features_t = self.hidden_layer_t[task_index](features_t)
        
        if self.version == 'v1':
            temp_alpha = F.softmax(self.alpha[task_index], 0)     # AMTL-v1,  alpha_1 + alpha_2 = 1
        elif self.version == 'v2':
            temp_alpha = torch.exp(self.alpha[task_index]) / (1 + torch.exp(self.alpha[task_index])) # AMTL-v2,  0 <= alpha <=1
        elif self.version == 'v3':
            # below for AMTL-v3, gumbel softmax
            temp = torch.sigmoid(self.alpha[task_index])
            if temp >= 0.5:
                temp_alpha = [0, 1]
            else:
                temp_alpha = [1, 0]
        else:
            print("No correct version parameter!")
            exit()

        hidden_features = temp_alpha[0] * hidden_features_s + temp_alpha[1] * hidden_features_t
        
        outputs = torch.mm(hidden_features, self.classifier_parameter[task_index])
        return outputs
        
    def get_adaptative_parameter(self):
        return self.alpha
        

class AMTL_new(nn.Module):
    def __init__(self, task_num, base_net='resnet50', hidden_dim=1024, class_num=31, version='v1'):
        super(AMTL_new, self).__init__()
        # shared base network
        self.base_network_s = resnet.__dict__[base_net](pretrained=True)
        # task-specific base network
        self.base_network_t = nn.ModuleList([resnet.__dict__[base_net](pretrained=True) for _ in range(task_num)])
        self.avgpool = self.base_network_s.avgpool
        # shared hidden layer
        self.hidden_layer_list_s = [nn.Linear(2048, hidden_dim),nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.hidden_layer_s = nn.Sequential(*self.hidden_layer_list_s)
        # task-specific hidden layer
        self.hidden_layer_list_t = [nn.ModuleList([nn.Linear(2048, hidden_dim),nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]) for _ in range(task_num)]
        self.hidden_layer_t = nn.ModuleList([nn.Sequential(*self.hidden_layer_list_t[t]) for t in range(task_num)])
        # shared classifier layer
        self.classifier_parameter_s = nn.Parameter(torch.FloatTensor(task_num, hidden_dim, class_num))
        # task-specific classifier layer
        self.classifier_parameter_t = nn.Parameter(torch.FloatTensor(task_num, hidden_dim, class_num))
        
        # initialization
        self.hidden_layer_s[0].weight.data.normal_(0, 0.005)
        self.hidden_layer_s[0].bias.data.fill_(0.1)
        self.classifier_parameter_s.data.normal_(0, 0.01)
        self.classifier_parameter_t.data.normal_(0, 0.01)
        for t in range(task_num):
            self.hidden_layer_t[t][0].weight.data.normal_(0, 0.005)
            self.hidden_layer_t[t][0].bias.data.fill_(0.1)
        
        self.version = version
        
        # adaptative parameters
        if self.version == 'v1' or self.version =='v2':
            # AMTL-v1 and v2
            self.alpha = nn.Parameter(torch.FloatTensor(task_num, 2))
            self.alpha.data.fill_(0.5)   # init 0.5(shared) 0.5(specific)
            # self.alpha.data[:,0].fill_(0)  # shared
            # self.alpha.data[:,1].fill_(1)  # specific
        elif self.version == 'v3':
            # AMTL-v3, gumbel softmax
            self.alpha = nn.Parameter(torch.FloatTensor(task_num))
            self.alpha.data.fill_(0)
        else:
            print("No correct version parameter!")
            exit()

    def forward(self, inputs, task_index):
        features_s = self.base_network_s(inputs)
        features_s = torch.flatten(self.avgpool(features_s), 1)
        hidden_features_s = self.hidden_layer_s(features_s)
        outputs_s = torch.mm(hidden_features_s, self.classifier_parameter_s[task_index])
        
        features_t = self.base_network_t[task_index](inputs)
        features_t = torch.flatten(self.avgpool(features_t), 1)
        hidden_features_t = self.hidden_layer_t[task_index](features_t)
        outputs_t = torch.mm(hidden_features_t, self.classifier_parameter_t[task_index])
        
        if self.version == 'v1':
            temp_alpha = F.softmax(self.alpha[task_index], 0)     # AMTL-v1,  alpha_1 + alpha_2 = 1
        elif self.version == 'v2':
            temp_alpha = torch.exp(self.alpha[task_index]) / (1 + torch.exp(self.alpha[task_index])) # AMTL-v2,  0 <= alpha <=1
        elif self.version == 'v3':
            # below for AMTL-v3, gumbel softmax
            temp = torch.sigmoid(self.alpha[task_index])
            temp_alpha = torch.stack([1-temp, temp])
            temp_alpha = F.gumbel_softmax(torch.log(temp_alpha), tau=0.1, hard=True)
        else:
            print("No correct version parameter!")
            exit()

        outputs = temp_alpha[0] * outputs_s + temp_alpha[1] * outputs_t
        
        return outputs
    
    def predict(self, inputs, task_index):
        features_s = self.base_network_s(inputs)
        features_s = torch.flatten(self.avgpool(features_s), 1)
        hidden_features_s = self.hidden_layer_s(features_s)
        outputs_s = torch.mm(hidden_features_s, self.classifier_parameter_s[task_index])
        
        features_t = self.base_network_t[task_index](inputs)
        features_t = torch.flatten(self.avgpool(features_t), 1)
        hidden_features_t = self.hidden_layer_t[task_index](features_t)
        outputs_t = torch.mm(hidden_features_t, self.classifier_parameter_t[task_index])
        
        if self.version == 'v1':
            temp_alpha = F.softmax(self.alpha[task_index], 0)     # AMTL-v1,  alpha_1 + alpha_2 = 1
        elif self.version == 'v2':
            temp_alpha = torch.exp(self.alpha[task_index]) / (1 + torch.exp(self.alpha[task_index])) # AMTL-v2,  0 <= alpha <=1
        elif self.version == 'v3':
            # below for AMTL-v3, gumbel softmax
            temp = torch.sigmoid(self.alpha[task_index])
            if temp >= 0.5:
                temp_alpha = [0, 1]
            else:
                temp_alpha = [1, 0]
        else:
            print("No correct version parameter!")
            exit()

        outputs = temp_alpha[0] * outputs_s + temp_alpha[1] * outputs_t
        
        return outputs
        
    def get_adaptative_parameter(self):
        return self.alpha

    
# other backbone
class Cross_Stitch(nn.Module):
    def __init__(self, task_num, base_net='resnet50', hidden_dim=1024, class_num=31):
        super(Cross_Stitch, self).__init__()
        self.task_num = task_num
        # base network
        self.base_network = nn.ModuleList([resnet.__dict__[base_net](pretrained=True) for _ in task_num])
                
        # We will apply the cross-stitch unit over the last bottleneck layer in the ResNet. 
        self.resnet_layer1 = nn.ModuleList([])
        self.resnet_layer2 = nn.ModuleList([])
        self.resnet_layer3 = nn.ModuleList([])
        self.resnet_layer4 = nn.ModuleList([])
        for i in range(self.task_num):
            self.resnet_layer1.append(base_network[i].layer1) 
            self.resnet_layer2.append(base_network[i].layer2)
            self.resnet_layer3.append(base_network[i].layer3)
            self.resnet_layer4.append(base_network[i].layer4)
        
        # define cross-stitch units
        self.cross_unit = nn.Parameter(data=torch.ones(4, self.task_num))
            
        # shared layer
        self.avgpool = self.base_network.avgpool
        self.hidden_layer_list = [nn.Linear(2048, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.hidden_layer = nn.Sequential(*self.hidden_layer_list)
                
        # task-specific layer
        self.classifier_parameter = nn.Parameter(torch.FloatTensor(self.task_num, hidden_dim, class_num))

        # initialization
        self.hidden_layer[0].weight.data.normal_(0, 0.005)
        self.hidden_layer[0].bias.data.fill_(0.1)
        self.classifier_parameter.data.normal_(0, 0.01)

    def forward(self, inputs, task_index):
        res_feature = [0 for _ in self.task_num]
        for j in range(self.task_num):
            res_feature[j] = [0, 0, 0, 0]
            
        features = self.base_network(inputs)
        features = torch.flatten(self.avgpool(features), 1)
        hidden_features = self.hidden_layer(features)
        outputs = torch.mm(hidden_features, self.classifier_parameter[task_index])
        return outputs

    def predict(self, inputs, task_index):
        return self.forward(inputs, task_index)