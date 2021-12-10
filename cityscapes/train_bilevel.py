import torch, time, os, random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
from backbone_bilevel import SMTLmodel, SMTLmodel_new
from utils import *

from create_dataset import  CityScape

import argparse

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description= 'SMTL on CityScapes')
    parser.add_argument('--gpu_id', default='0', help='gpu_id') 
    parser.add_argument('--model', default='SMTL', type=str, help='SMTL, SMTL_new')
    parser.add_argument('--aug', type=str, default='False', help='data augmentation')
    parser.add_argument('--train_mode', default='train', type=str, help='trainval, train')
    parser.add_argument('--total_epoch', default=200, type=int, help='training epoch')
    # for SMTL
    parser.add_argument('--version', default='v1', type=str, help='v1 (a1+a2=1), v2 (0<=a<=1), v3 (gumbel softmax)')
    return parser.parse_args()

params = parse_args()
print(params)


os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

dataset_path = '/data/dataset/cityscapes2/'


def build_model():
    if params.model == 'SMTL':
        batch_size = 20
        model = SMTLmodel(version=params.version).cuda()
    elif params.model == 'SMTL_new':
        batch_size = 70
        model = SMTLmodel_new(version=params.version).cuda()
    else:
        print("No correct model parameter!")
        exit()
    return model, batch_size

model, batch_size = build_model()
    
task_num = len(model.tasks)
    
cityscapes_train_set = CityScape(root=dataset_path, mode=params.train_mode, augmentation=params.aug)
cityscapes_val_set = CityScape(root=dataset_path, mode='val', augmentation=params.aug)
cityscapes_test_set = CityScape(root=dataset_path, mode='test', augmentation='False')

cityscapes_train_loader = torch.utils.data.DataLoader(
    dataset=cityscapes_train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True)
    
cityscapes_val_loader = torch.utils.data.DataLoader(
    dataset=cityscapes_val_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True)

cityscapes_test_loader = torch.utils.data.DataLoader(
    dataset=cityscapes_test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# gate parameter, alpha
class Model_alpha(nn.Module):
    def __init__(self, task_num=2, version='v1'):
        super(Model_alpha, self).__init__()
        # adaptative parameters
        if version == 'v1' or version =='v2':
            # SMTL-v1 and v2
            self.alpha = nn.Parameter(torch.FloatTensor(task_num, 2))
            self.alpha.data.fill_(0.5)   # init 0.5(shared) 0.5(specific)
            # self.alpha.data[:,0].fill_(0)  # shared
            # self.alpha.data[:,1].fill_(1)  # specific
        elif version == 'v3':
            # SMTL-v3, gumbel softmax
            self.alpha = nn.Parameter(torch.FloatTensor(task_num))
            self.alpha.data.fill_(0)
        else:
            print("No correct version parameter!")
            exit()
    
    def get_adaptative_parameter(self):
        return self.alpha

h = Model_alpha(task_num=task_num, version=params.version).cuda()
h.train()
h_optimizer = torch.optim.Adam(h.parameters(), lr=1e-4)


def set_param(curr_mod, name, param=None, mode='update'):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode)
    else:
        if mode == 'update':
            delattr(curr_mod, name)
            setattr(curr_mod, name, param)
        elif mode == 'get':
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p
                

print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR')
total_epoch = params.total_epoch
train_batch = len(cityscapes_train_loader)
val_batch = len(cityscapes_val_loader)
avg_cost = torch.zeros([total_epoch, 24])
lambda_weight = torch.ones([task_num, total_epoch]).cuda()
for index in range(total_epoch):
    s_t = time.time()
    cost = torch.zeros(24)
    # iteration for all batches
    model.train()
    train_dataset = iter(cityscapes_train_loader)
    val_dataset = iter(cityscapes_val_loader)
    conf_mat = ConfMatrix(model.class_nb)
    for k in range(min(train_batch, val_batch)):
        meta_model, _ = build_model()
        meta_model.load_state_dict(model.state_dict())
        
        model_np = {}
        for n, p in meta_model.named_parameters():
            model_np[n] = p
            
        train_data, train_label, train_depth = train_dataset.next()
        train_data, train_label = train_data.cuda(non_blocking=True), train_label.long().cuda(non_blocking=True)
        train_depth = train_depth.cuda(non_blocking=True)
        train_pred = meta_model(train_data, h)

        train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                      model_fit(train_pred[1], train_depth, 'depth')]
        loss_train = torch.zeros(2).cuda()
        for i in range(2):
            loss_train[i] = train_loss[i]
        loss_sum = torch.sum(loss_train*lambda_weight[:, index])
        
        meta_model.zero_grad()
        grads = torch.autograd.grad(loss_sum, (meta_model.parameters()), create_graph=True)
        
        for g_index, name in enumerate(model_np.keys()):
            p = set_param(meta_model, name, mode='get')
            if grads[g_index] == None:
                print(g_index, name, grads[g_index])
                continue
            p_fast = p - 1e-4 * grads[g_index]
            set_param(meta_model, name, param=p_fast, mode='update')
            model_np[name] = p_fast
        del grads
        del model_np
        
        # update outer loop
        val_data, val_label, val_depth = val_dataset.next()
        val_data, val_label = val_data.cuda(non_blocking=True), val_label.long().cuda(non_blocking=True)
        val_depth = val_depth.cuda(non_blocking=True)
        valid_pred = meta_model(val_data, h)
        valid_loss = [model_fit(valid_pred[0], val_label, 'semantic'),
                      model_fit(valid_pred[1], val_depth, 'depth')]
        loss_val = torch.zeros(2).cuda()
        for i in range(2):
            loss_val[i] = valid_loss[i]
        loss_all = torch.sum(loss_val*lambda_weight[:, index])
        h_optimizer.zero_grad()
        loss_all.backward()
        h_optimizer.step()
        
        # update inner loop
        train_pred = model(train_data, h)
        train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                      model_fit(train_pred[1], train_depth, 'depth')]
        loss_final = torch.zeros(2).cuda()
        for i in range(2):
            loss_final[i] = train_loss[i]
        loss = torch.sum(loss_final*lambda_weight[:, index])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulate label prediction for every pixel in training images
        conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

        cost[0] = train_loss[0].item()
        cost[3] = train_loss[1].item()
        cost[4], cost[5] = depth_error(train_pred[1], train_depth)
        avg_cost[index, :12] += cost[:12] / train_batch

    # compute mIoU and acc
    avg_cost[index, 1], avg_cost[index, 2] = conf_mat.get_metrics()

    # evaluating test data
    model.eval()
    conf_mat = ConfMatrix(model.class_nb)
    with torch.no_grad():  # operations inside don't track history
        test_dataset = iter(cityscapes_test_loader)
        test_batch = len(cityscapes_test_loader)
        for k in range(test_batch):
            test_data, test_label, test_depth = test_dataset.next()
            test_data, test_label = test_data.cuda(non_blocking=True), test_label.long().cuda(non_blocking=True)
            test_depth = test_depth.cuda(non_blocking=True)
            test_pred = model.predict(test_data, h)
            test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                         model_fit(test_pred[1], test_depth, 'depth')]

            conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

            cost[12] = test_loss[0].item()
            cost[15] = test_loss[1].item()
            cost[16], cost[17] = depth_error(test_pred[1], test_depth)
            
            avg_cost[index, 12:] += cost[12:] / test_batch

        # compute mIoU and acc
        avg_cost[index, 13], avg_cost[index, 14] = conf_mat.get_metrics()
    
    scheduler.step()
    e_t = time.time()
    if params.model == 'SMTL' or params.model == 'SMTL_new':
        alpha = h.get_adaptative_parameter()
        for i in range(task_num):
            if params.version == 'v1':
                print(alpha[i], F.softmax(alpha[i], 0))   # SMTL-v1, alpha_1 + alpha_2 = 1
            elif params.version == 'v2':
                print(alpha[i], torch.exp(alpha[i]) / (1 + torch.exp(alpha[i])))  # SMTL-v2, 0 <= alpha <= 1
            elif params.version == 'v3':
                # below for SMTL-v3, gumbel softmax
                temp = torch.sigmoid(alpha[i])
                temp_alpha = torch.stack([1-temp, temp])
                print(i, temp_alpha)
            else:
                print("No correct version parameter!")
                exit()
    print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} ||'
        'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} || {:.4f}'
        .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 12], avg_cost[index, 13],
                avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], e_t-s_t))