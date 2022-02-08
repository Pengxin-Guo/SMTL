import torch, time, os, random, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
from backbone import SMTLmodel_weight
from utils import *

from create_dataset import NYUv2

from weighting_utils import weight_update
from min_norm_solvers import MinNormSolver, gradient_normalizers

import argparse

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description= 'SMTL for NYUv2')
    parser.add_argument('--gpu_id', default='0', help='gpu_id') 
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_mode', default='trainval', type=str, help='trainval, train')
    parser.add_argument('--random_distribution', default='normal', type=str, 
                        help='normal, random_normal, uniform, inter_random, dirichlet, dropout, dropout_1, dropout_2')
    parser.add_argument('--weighting', default='EW', type=str, 
                        help='EW, UW, DWA, random, GLS')
    # for SMTL
    parser.add_argument('--version', default='v1', type=str, help='v1 (a1+a2=1), v2 (0<=a<=1), v3 (gumbel softmax)')
    return parser.parse_args()

params = parse_args()
print(params)

'''
save_weight_path = './weight/{}_{}'.format(params.model, params.weighting)
if params.weighting == 'MGDA':
    save_weight_path += '_{}'.format(params.mgda_gn)
elif params.weighting == 'random':
    save_weight_path += '_{}'.format(params.random_distribution)
if params.aug:
    save_weight_path += '_aug'
save_weight_path += '.mat'
'''

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

batch_size = 4

dataset_path = '/data/dataset/nyuv2/'
nyuv2_train_set = NYUv2(root=dataset_path, mode=params.train_mode, augmentation=params.aug)
nyuv2_test_set = NYUv2(root=dataset_path, mode='test', augmentation=False)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True)

nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True)

model = SMTLmodel_weight(version=params.version, weighting=params.weighting).cuda()
task_num = len(model.tasks)
scheduler = None
init_loss = None

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

mean, std = None, None
if params.random_distribution == 'random_normal':
    mean, std = torch.rand(task_num), torch.rand(task_num)
    
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')
total_epoch = 200
train_batch = len(nyuv2_train_loader)
avg_cost = torch.zeros([total_epoch, 24])
lambda_weight = torch.ones([task_num, total_epoch, train_batch]).cuda()
for epoch in range(total_epoch):
    s_t = time.time()
    cost = torch.zeros(24)

    # iteration for all batches
    model.train()
    train_dataset = iter(nyuv2_train_loader)
    conf_mat = ConfMatrix(model.class_nb)
    for batch_index in range(train_batch):
#         if batch_index > 1:
#             break
        train_data, train_label, train_depth, train_normal = train_dataset.next()
        train_data, train_label = train_data.cuda(non_blocking=True), train_label.long().cuda(non_blocking=True)
        train_depth, train_normal = train_depth.cuda(non_blocking=True), train_normal.cuda(non_blocking=True)
        
        train_pred = model.forward(train_data)

        train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                      model_fit(train_pred[1], train_depth, 'depth'),
                      model_fit(train_pred[2], train_normal, 'normal')]
        loss_train = torch.zeros(3).cuda()
        for i in range(3):
            loss_train[i] = train_loss[i]
            
        batch_weight = weight_update(params.weighting, loss_train, model, optimizer, epoch, 
                                     batch_index, task_num, clip_grad=False, scheduler=None, 
                                     random_distribution=params.random_distribution, 
                                     avg_cost=avg_cost[:,0:7:3], mean=mean, std=std, init_loss=init_loss)
        if batch_weight is not None:
            lambda_weight[:, epoch, batch_index] = batch_weight
        
        # accumulate label prediction for every pixel in training images
        conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

        cost[0] = train_loss[0].item()
        cost[3] = train_loss[1].item()
        cost[4], cost[5] = depth_error(train_pred[1], train_depth)
        cost[6] = train_loss[2].item()
        cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
        avg_cost[epoch, :12] += cost[:12] / train_batch

    # compute mIoU and acc
    avg_cost[epoch, 1], avg_cost[epoch, 2] = conf_mat.get_metrics()

    # evaluating test data
    model.eval()
    conf_mat = ConfMatrix(model.class_nb)
    with torch.no_grad():  # operations inside don't track history
        val_dataset = iter(nyuv2_test_loader)
        val_batch = len(nyuv2_test_loader)
        for k in range(val_batch):
            val_data, val_label, val_depth, val_normal = val_dataset.next()
            val_data, val_label = val_data.cuda(non_blocking=True), val_label.long().cuda(non_blocking=True)
            val_depth, val_normal = val_depth.cuda(non_blocking=True), val_normal.cuda(non_blocking=True)

            val_pred = model(val_data)
            val_loss = [model_fit(val_pred[0], val_label, 'semantic'),
                         model_fit(val_pred[1], val_depth, 'depth'),
                         model_fit(val_pred[2], val_normal, 'normal')]

            conf_mat.update(val_pred[0].argmax(1).flatten(), val_label.flatten())

            cost[12] = val_loss[0].item()
            cost[15] = val_loss[1].item()
            cost[16], cost[17] = depth_error(val_pred[1], val_depth)
            cost[18] = val_loss[2].item()
            cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(val_pred[2], val_normal)
            avg_cost[epoch, 12:] += cost[12:] / val_batch

        # compute mIoU and acc
        avg_cost[epoch, 13], avg_cost[epoch, 14] = conf_mat.get_metrics()
    
    scheduler.step()
    e_t = time.time()
    print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ||'
        'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} || {:.4f}'
        .format(epoch, avg_cost[epoch, 0], avg_cost[epoch, 1], avg_cost[epoch, 2], avg_cost[epoch, 3],
                avg_cost[epoch, 4], avg_cost[epoch, 5], avg_cost[epoch, 6], avg_cost[epoch, 7], avg_cost[epoch, 8],
                avg_cost[epoch, 9], avg_cost[epoch, 10], avg_cost[epoch, 11], avg_cost[epoch, 12], avg_cost[epoch, 13],
                avg_cost[epoch, 14], avg_cost[epoch, 15], avg_cost[epoch, 16], avg_cost[epoch, 17], avg_cost[epoch, 18],
                avg_cost[epoch, 19], avg_cost[epoch, 20], avg_cost[epoch, 21], avg_cost[epoch, 22], avg_cost[epoch, 23], e_t-s_t))
    
# sio.savemat(save_weight_path, {'weight':lambda_weight.cpu().numpy()})