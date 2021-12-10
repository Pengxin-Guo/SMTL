import torch, time, os, random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
from backbone import DeepLabv3, Cross_Stitch, MTANDeepLabv3, AdaShare, SMTLmodel, SMTLmodel_new
from nddr_cnn import NDDRCNN
from afa import AFANet
from utils import *

from create_dataset import  CityScape

import argparse

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description= 'SMTL on CityScapes')
    parser.add_argument('--gpu_id', default='0', help='gpu_id') 
    parser.add_argument('--model', default='DMTL', type=str, help='DMTL, CROSS, MTAN, AdaShare, NDDRCNN, AFA, SMTL, SMTL_new')
    parser.add_argument('--aug', type=str, default='False', help='data augmentation')
    parser.add_argument('--train_mode', default='trainval', type=str, help='trainval, train')
    parser.add_argument('--total_epoch', default=200, type=int, help='training epoch')
    # for SMTL
    parser.add_argument('--version', default='v1', type=str, help='v1 (a1+a2=1), v2 (0<=a<=1), v3 (gumbel softmax)')
    return parser.parse_args()

params = parse_args()
print(params)


os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

dataset_path = '/data/dataset/cityscapes2/'
if params.model == 'DMTL':
    batch_size = 180
    model = DeepLabv3().cuda()
elif params.model == 'CROSS':
    batch_size = 100
    model = Cross_Stitch().cuda()
elif params.model == 'MTAN':
    batch_size = 80
    model = MTANDeepLabv3().cuda()
elif params.model == 'AdaShare':
    batch_size = 120
    model = AdaShare().cuda()
elif params.model == 'NDDRCNN':
    batch_size = 80
    model = NDDRCNN().cuda()
elif params.model == 'AFA':
    batch_size = 150
    model = AFANet().cuda()
elif params.model == 'SMTL':
    batch_size = 70
    model = SMTLmodel(version=params.version).cuda()
elif params.model == 'SMTL_new':
    batch_size = 70
    model = SMTLmodel_new(version=params.version).cuda()
else:
    print("No correct model parameter!")
    exit()
    
task_num = len(model.tasks)
    
cityscapes_train_set = CityScape(root=dataset_path, mode=params.train_mode, augmentation=params.aug)
cityscapes_test_set = CityScape(root=dataset_path, mode='test', augmentation='False')

cityscapes_train_loader = torch.utils.data.DataLoader(
    dataset=cityscapes_train_set,
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

print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR')
total_epoch = params.total_epoch
train_batch = len(cityscapes_train_loader)
avg_cost = torch.zeros([total_epoch, 24])
lambda_weight = torch.ones([task_num, total_epoch]).cuda()
for index in range(total_epoch):
    s_t = time.time()
    cost = torch.zeros(24)

    # iteration for all batches
    model.train()
    train_dataset = iter(cityscapes_train_loader)
    conf_mat = ConfMatrix(model.class_nb)
    for k in range(train_batch):
        train_data, train_label, train_depth = train_dataset.next()
        train_data, train_label = train_data.cuda(non_blocking=True), train_label.long().cuda(non_blocking=True)
        train_depth = train_depth.cuda(non_blocking=True)

        train_pred = model(train_data)

        train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                      model_fit(train_pred[1], train_depth, 'depth')]
        loss_train = torch.zeros(2).cuda()
        for i in range(2):
            loss_train[i] = train_loss[i]
        
        loss = torch.sum(loss_train*lambda_weight[:, index])
           
        # for single task
        # loss = loss_train[1]
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
        val_dataset = iter(cityscapes_test_loader)
        val_batch = len(cityscapes_test_loader)
        for k in range(val_batch):
            val_data, val_label, val_depth = val_dataset.next()
            val_data, val_label = val_data.cuda(non_blocking=True), val_label.long().cuda(non_blocking=True)
            val_depth = val_depth.cuda(non_blocking=True)
            val_pred = model.predict(val_data)
            val_loss = [model_fit(val_pred[0], val_label, 'semantic'),
                         model_fit(val_pred[1], val_depth, 'depth')]

            conf_mat.update(val_pred[0].argmax(1).flatten(), val_label.flatten())

            cost[12] = val_loss[0].item()
            cost[15] = val_loss[1].item()
            cost[16], cost[17] = depth_error(val_pred[1], val_depth)
            
            avg_cost[index, 12:] += cost[12:] / val_batch

        # compute mIoU and acc
        avg_cost[index, 13], avg_cost[index, 14] = conf_mat.get_metrics()
    
    scheduler.step()
    e_t = time.time()
    if params.model == 'SMTL' or params.model == 'SMTL_new':
        alpha = model.get_adaptative_parameter()
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