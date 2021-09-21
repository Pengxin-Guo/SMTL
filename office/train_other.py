import torch, time, os, random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from backbone import Cross_Stitch
from create_dataset import office_dataloader_other
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
    parser.add_argument('--model', default='Cross', type=str, help='Cross')
    parser.add_argument('--train_mode', default='trval', type=str, help='trval, train')
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

if params.model == 'Cross':
    batchsize = 32
    model = Cross_Stitch(task_num=task_num, class_num=class_num).cuda()
elif params.model == 'MTAN':
    batchsize = 32
    model = MTAN_ResNet(task_num, class_num).cuda()
else:
    print("No correct model parameter!")
    exit()
    
data_loader, iter_data_loader = office_dataloader_other(params.dataset, batchsize=batchsize)

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
        train_datas = [0 for _ in range(task_num)]
        train_labels=  [0 for _ in range(task_num)]
        for task_index in range(task_num):
            try:
                train_data, train_label = iter_data_loader[task_index][params.train_mode].next()
            except:
                iter_data_loader[task_index][params.train_mode] = iter(data_loader[task_index][params.train_mode])
                train_data, train_label = iter_data_loader[task_index][params.train_mode].next()
            train_data, train_label = train_data.cuda(non_blocking=True), train_label.cuda(non_blocking=True)
            
            train_datas[task_index] = train_data
            train_labels[task_index] = train_label
            
        outputs = model(train_datas)
        for task_index in range(task_num):
            loss_train[task_index] = loss_fn(outputs[task_index], train_labels[task_index])
            avg_cost[epoch, task_index] += loss_train[task_index].item()
        
        loss = torch.sum(loss_train*lambda_weight[:, epoch])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_cost[epoch] /= train_batch

    # evaluating test data
    model.eval()
    # wrong !!!
    with torch.no_grad(): 
        right_num = np.zeros([task_num])
        count = np.zeros([task_num])
        loss_data_count = np.zeros([task_num])
        # for mode_index, mode in enumerate(['val', 'test']):
        for mode_index, mode in enumerate(['test']):
            x_tests = [0 for _ in range(task_num)]
            y_tests = [0 for _ in range(task_num)]
            for k in range(task_num):
                for test_it, test_data in enumerate(data_loader[k][mode]):
                    x_test, y_test = test_data[0].cuda(non_blocking=True), test_data[1].cuda(non_blocking=True)
                    x_tests[k] = x_test
                    y_tests[k] = y_test
                    
            y_preds = model.predict(x_tests)
            for k in range(task_num):
                loss_t = loss_fn(y_preds[k], y_tests[k])
                loss_data_count[k] += loss_t.item()
                right_num[k] += ((torch.max(F.softmax(y_preds[k], dim=-1), dim=-1)[1])==y_tests[k]).sum().item()
                count[k] += y_tests[k].shape[0]
        acc_avg = (right_num/count).mean(axis=-1)
        loss_data_avg = (loss_data_count/count).mean(axis=-1)
        
        # print('val acc {} {}, loss {}'.format(right_num[0]/count[0], acc_avg[0], loss_data_count[0]))
        print('test acc {} {}, loss {}'.format(right_num/count, acc_avg, loss_data_count))
    e_t = time.time()
    print('-- cost time {}'.format(e_t-s_t))

    if acc_avg > best_test_acc:
        best_test_acc = acc_avg
        print('!! -- -- epoch {}; best test acc {} {}'.format(epoch, right_num/count, acc_avg))