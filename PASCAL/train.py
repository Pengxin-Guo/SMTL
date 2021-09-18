import torch, time, os, random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from data.pascal_context import PASCALContext
from data.custom_collate import collate_mil
from loss_functions import get_loss
from evaluation.evaluate_utils import PerformanceMeter, get_output

from backbone import DeepLabv3, Cross_Stitch, MTANDeepLabv3, AdaShare, AMTLmodel, AMTLmodel_new
from nddr_cnn import NDDRCNN
from afa import AFANet
import argparse

torch.set_num_threads(2)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description= 'AMTL for PASCAL')
    parser.add_argument('--task_index', default=8, type=int, help='for STL: 0,1,2,3')
    parser.add_argument('--gpu_id', default='0', help='gpu_id') 
    parser.add_argument('--total_epoch', default=200, type=int, help='training epoch')
    parser.add_argument('--model', default='DMTL', type=str, help='DMTL, CROSS, MTAN, AdaShare, NDDRCNN, AFA, AMTL, AMTL_new')
    # for AMTL
    parser.add_argument('--version', default='v1', type=str, help='v1 (a1+a2=1), v2 (0<=a<=1), v3 (gumbel softmax)')
    return parser.parse_args()

params = parse_args()
print(params)

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

def adjust_learning_rate(optimizer, epoch, total_epoch=60):
    lr = 1e-4
    lambd = pow(1-(epoch/total_epoch), 0.9)
    lr = lr * lambd
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

tasks = ['semseg', 'human_parts', 'sal', 'normals']
total_epoch = params.total_epoch

if params.task_index < len(tasks):
    tasks = [tasks[params.task_index]]


if params.model == 'DMTL':
    batch_size = 40
    model = DeepLabv3(tasks=tasks).cuda()
elif params.model == 'CROSS':
    batch_size = 24
    model = Cross_Stitch(tasks=tasks).cuda()
elif params.model == 'MTAN':
    batch_size = 20
    model = MTANDeepLabv3(tasks=tasks).cuda()
elif params.model == 'AdaShare':
    batch_size = 32
    model = AdaShare(tasks=tasks).cuda()
elif params.model == 'NDDRCNN':
    batch_size = 18
    model = NDDRCNN(tasks=tasks).cuda()
elif params.model == 'AFA':
    batch_size = 25
    model = AFANet(tasks=tasks).cuda()
elif params.model == 'AMTL':
    batch_size = 18
    model = AMTLmodel(tasks=tasks, version=params.version).cuda()
elif params.model == 'AMTL_new':
    batch_size = 15
    model = AMTLmodel_new(tasks=tasks, version=params.version).cuda()
else:
    print("No correct model parameter!")
    exit()

train_database = PASCALContext(split=['train'], aug=True,
                               do_edge='edge' in tasks,
                               do_human_parts='human_parts' in tasks,
                               do_semseg='semseg' in tasks,
                               do_normals='normals' in tasks,
                               do_sal='sal' in tasks)
test_database = PASCALContext(split=['val'], aug=False,
                              do_edge='edge' in tasks,
                              do_human_parts='human_parts' in tasks,
                              do_semseg='semseg' in tasks,
                              do_normals='normals' in tasks,
                              do_sal='sal' in tasks)

trainloader = DataLoader(train_database, batch_size=batch_size, shuffle=True, drop_last=True,
                 num_workers=4, collate_fn=collate_mil)
testloader = DataLoader(test_database, batch_size=batch_size, shuffle=False, drop_last=False,
                 num_workers=4)

task_num = len(model.tasks)

criterion = {task: get_loss(task).cuda() for task in tasks}

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
train_batch = len(trainloader)
avg_cost = torch.zeros([total_epoch, 2*task_num])
for epoch in range(total_epoch):
    print('-'*10, epoch)
    s_t = time.time()
    
    adjust_learning_rate(optimizer, epoch, total_epoch)
    
    # iteration for all batches
    model.train()
    train_dataset = iter(trainloader)
    performance_meter = PerformanceMeter(tasks)
    for batch_index in range(train_batch):
#         if batch_index > 5:
#             break
        
        train_batch_data = train_dataset.next()
        train_data = train_batch_data['image'].cuda(non_blocking=True)
        targets = {task: train_batch_data[task].cuda(non_blocking=True) for task in tasks}
        
        train_pred = model(train_data)

        loss_train = torch.zeros(task_num).cuda()
        for tk, task in enumerate(tasks):
            loss_train[tk] = criterion[task](train_pred[task], targets[task])
            avg_cost[epoch, tk] += loss_train[tk].item()

        optimizer.zero_grad()
        (loss_train.sum()).backward()
        optimizer.step()
            
        performance_meter.update({t: get_output(train_pred[t], t) for t in tasks}, 
                                 {t: targets[t] for t in tasks})
    
    eval_results_train = performance_meter.get_score(verbose=False)
    if params.model == 'AMTL' or params.model == 'AMTL_new':
        alpha = model.get_adaptative_parameter()
        for i in range(task_num):
            if params.version == 'v1':
                print(alpha[i], F.softmax(alpha[i], 0))   # AMTL-v1, alpha_1 + alpha_2 = 1
            elif params.version == 'v2':
                print(alpha[i], torch.exp(alpha[i]) / (1 + torch.exp(alpha[i])))  # AMTL-v2, 0 <= alpha <= 1
            elif params.version == 'v3':
                # below for AMTL-v3, gumbel softmax
                temp = torch.sigmoid(alpha[i])
                temp_alpha = torch.stack([1-temp, temp])
                print(i, temp_alpha)
            else:
                print("No correct version parameter!")
                exit()   
    print('TRAIN:', eval_results_train)
    avg_cost[epoch, :task_num] /= train_batch
        

    # evaluating test data
    model.eval()
    with torch.no_grad():  # operations inside don't track history
        val_dataset = iter(testloader)
        val_batch = len(testloader)
        performance_meter = PerformanceMeter(tasks)
        for k in range(val_batch):
#             if k > 5:
#                 break
            val_batch_data = val_dataset.next()
            val_data = val_batch_data['image'].cuda(non_blocking=True)
            targets = {task: val_batch_data[task].cuda(non_blocking=True) for task in tasks}

            val_pred = model.predict(val_data)
            for tk, task in enumerate(tasks):
                avg_cost[epoch, task_num+tk] += (criterion[task](val_pred[task], targets[task])).item()
            performance_meter.update({t: get_output(val_pred[t], t) for t in tasks}, 
                                 {t: targets[t] for t in tasks})
        eval_results_test = performance_meter.get_score(verbose=False)
        print('TEST:', eval_results_test)
        avg_cost[epoch, task_num:] /= val_batch

    e_t = time.time()
    print('TIME:', e_t-s_t)
