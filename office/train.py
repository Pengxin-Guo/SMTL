import torch, time, os, random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from backbone import MTAN_ResNet, DMTL, AMTL, AMTL_new
from create_dataset import office_dataloader
import argparse
torch.set_num_threads(3)

torch.manual_seed(688)
random.seed(688)
np.random.seed(688)

def parse_args():
    parser = argparse.ArgumentParser(description= 'AMTL for Office-31 and Office-Home')
    parser.add_argument('--dataset', default='office-31', type=str, help='office-31, office-home')
    parser.add_argument('--task_index', default=10, type=int, help='for STL: 0,1,2,3')
    parser.add_argument('--gpu_id', default='0', help='gpu_id') 
    parser.add_argument('--model', default='DMTL', type=str, help='DMTL, MTAN, AMTL, AMTL_new')
    parser.add_argument('--train_mode', default='trval', type=str, help='trval, train')
    # for AMTL
    parser.add_argument('--version', default='v1', type=str, help='v1 (a1+a2=1), v2 (0<=a<=1), v3 (gumbel softmax)')
    return parser.parse_args()

params = parse_args()
print(params)

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

if params.dataset == 'office-31':
    task_num, class_num = 3, 31
elif params.dataset == 'office-home':
    task_num, class_num = 4, 65
else:
    print("No correct dataset parameter!")
    exit()

if params.model == 'DMTL':
    batchsize = 64
    model = DMTL(task_num=task_num, class_num=class_num).cuda()
elif params.model == 'MTAN':
    batchsize = 32
    model = MTAN_ResNet(task_num, class_num).cuda()
elif params.model == 'AMTL':
    batchsize = 32
    model = AMTL(task_num=task_num, class_num=class_num, version=params.version).cuda()
elif params.model == 'AMTL_new':
    batchsize = 32
    model = AMTL_new(task_num=task_num, class_num=class_num, version=params.version).cuda()
else:
    print("No correct model parameter!")
    exit()
    
data_loader, iter_data_loader = office_dataloader(params.dataset, batchsize=batchsize)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

total_epoch = 50
train_batch = max(len(data_loader[i][params.train_mode]) for i in range(task_num))
avg_cost = torch.zeros([total_epoch, task_num])
lambda_weight = torch.ones([task_num, total_epoch]).cuda()
loss_fn = nn.CrossEntropyLoss().cuda()
best_test_acc = 0
for epoch in range(total_epoch):
    print('--- Epoch {}'.format(epoch))
    s_t = time.time()
    model.train()
    for batch_index in range(train_batch):
        loss_train = torch.zeros(task_num).cuda()
        for task_index in range(task_num):
            try:
                train_data, train_label = iter_data_loader[task_index][params.train_mode].next()
            except:
                iter_data_loader[task_index][params.train_mode] = iter(data_loader[task_index][params.train_mode])
                train_data, train_label = iter_data_loader[task_index][params.train_mode].next()
            train_data, train_label = train_data.cuda(non_blocking=True), train_label.cuda(non_blocking=True)
            loss_train[task_index] = loss_fn(model(train_data, task_index), train_label)
            avg_cost[epoch, task_index] += loss_train[task_index].item()
        
        if params.task_index > task_num:    
            loss = torch.sum(loss_train*lambda_weight[:, epoch])
        else:
            loss = loss_train[params.task_index]   # for STL
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_cost[epoch] /= train_batch

    # evaluating test data
    model.eval()
    with torch.no_grad(): 
        right_num = np.zeros([2, task_num])
        count = np.zeros([2, task_num])
        loss_data_count = np.zeros([2, task_num])
        for mode_index, mode in enumerate(['val', 'test']):
            for k in range(task_num):
                for test_it, test_data in enumerate(data_loader[k][mode]):
                    x_test, y_test = test_data[0].cuda(non_blocking=True), test_data[1].cuda(non_blocking=True)
                    y_pred = model.predict(x_test, k)
                    loss_t = loss_fn(y_pred, y_test)
                    loss_data_count[mode_index, k] += loss_t.item()
                    right_num[mode_index, k] += ((torch.max(F.softmax(y_pred, dim=-1), dim=-1)[1])==y_test).sum().item()
                    count[mode_index, k] += y_test.shape[0]
        acc_avg = (right_num/count).mean(axis=-1)
        loss_data_avg = (loss_data_count/count).mean(axis=-1)
        
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
        
        # print('val acc {} {}, loss {}'.format(right_num[0]/count[0], acc_avg[0], loss_data_count[0]))
        print('test acc {} {}, loss {}'.format(right_num[1]/count[1], acc_avg[1], loss_data_count[1]))
    e_t = time.time()
    print('-- cost time {}'.format(e_t-s_t))
    
    if params.task_index > task_num:
        if acc_avg[1] > best_test_acc:
            best_test_acc = acc_avg[1]
            print('!! -- -- epoch {}; best test acc {} {}'.format(epoch, right_num[1]/count[1], acc_avg[1]))
    else:
        # for single task
        task_index = task_index
        if (right_num[1]/count[1])[task_index] > best_test_acc:
            best_test_acc = (right_num[1]/count[1])[task_index]
            print('!! -- -- epoch {}; best test acc {}'.format(epoch, right_num[1]/count[1]))