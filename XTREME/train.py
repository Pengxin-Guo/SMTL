import random, os, time, argparse, sys, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup, logging
logging.set_verbosity_error()
logging.set_verbosity_warning()
from tqdm import tqdm

from create_dataset import DataloaderSC, DataloaderTC
from model import *
from utils import get_data, get_metric
from torch.utils.tensorboard import SummaryWriter
from utils import weight_update

'''
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
'''

def parse_args():
    parser = argparse.ArgumentParser(description= 'SMTL for multilingual tasks')
    parser.add_argument('--dataset', default='udpos', type=str, help='xnli, pawsx, panx, udpos')
    parser.add_argument('--gpu_id', default='1', help='gpu_id') 
    parser.add_argument('--model', default='DMTL', type=str, help='DMTL, STL, SMTL, SMTL_new')
    parser.add_argument('--lang', default='all', type=str, help='all, en, zh, te, vi, de, es')
    parser.add_argument('--name', default='', type=str, help='name')
    # for SMTL
    parser.add_argument('--version', default='v1', type=str, help='v1 (a1+a2=1), v2 (0<=a<=1), v3 (gumbel softmax)')
    return parser.parse_args()

params = parse_args()
print(params)

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id

if params.dataset == 'udpos':
    lang_list = ['en', 'zh', 'te', 'vi']
else:
    lang_list = ['en', 'zh', 'de', 'es']

if params.model == 'STL':
    lang_list = [params.lang]
task_num = len(lang_list)
model_name_or_path = 'bert-base-multilingual-cased'
model_type = 'bert'
mode_list = ['train', 'dev', 'test']
max_seq_length = 128

root_data = '/data/dataset/XTREME/'

if params.model == 'STL':
    batch_size = 32
elif params.model == 'DMTL':
    batch_size = 32
elif params.model == 'SMTL':
    batch_size = 16
elif params.model == 'SMTL_new':
    batch_size = 16
else:
    raise('No support model!')

if params.dataset in ['xnli', 'pawsx']:
    task_type = 'SC'
    data_dir = '{}/{}'.format(root_data, params.dataset)
    dataloader, iter_dataloader, labels = DataloaderSC(lang_list=lang_list,
                                                  model_name_or_path=model_name_or_path,
                                                  model_type=model_type,
                                                  mode_list=mode_list,
                                                  data_dir=data_dir,
                                                  max_seq_length=max_seq_length,
                                                  batch_size=batch_size)
elif params.dataset in ['panx', 'udpos']:
    task_type = 'TC'
    data_dir = '{}/{}/{}_processed_maxlen128/'.format(root_data, params.dataset, params.dataset)
    dataloader, iter_dataloader, labels = DataloaderTC(lang_list=lang_list,
                                                  model_name_or_path=model_name_or_path,
                                                  model_type=model_type,
                                                  mode_list=mode_list,
                                                  data_dir=data_dir,
                                                  max_seq_length=max_seq_length,
                                                  batch_size=batch_size)
else:
    raise('No support dataset!')
    
if params.model == 'STL':
    lang_list = [params.lang]
    task_num = len(lang_list)
    model = STL(label_num=len(labels), task_num=task_num, task_type=task_type).cuda()
elif params.model == 'DMTL':
    model = mBert(label_num=len(labels), task_num=task_num, task_type=task_type).cuda()
elif params.model == 'SMTL':
    model = SMTL_mBert(label_num=len(labels), task_num=task_num, task_type=task_type, version=params.version).cuda()
elif params.model == 'SMTL_new':
    model = SMTL_new_mBert(label_num=len(labels), task_num=task_num, task_type=task_type, version=params.version).cuda()
else:
    print("No support model!")
    exit()


'''
logfolder = params.model + "_" + params.dataset + "_" + params.lang
# logfolder = "debug"
logdir = os.path.join('./writer', logfolder)
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir=logdir)
print(logdir)
'''

total_epoch = 300
train_batch = max(len(dataloader[lg]['train']) for lg in lang_list)
t_total = train_batch*total_epoch

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0, 
                                            num_training_steps=t_total)

best_dev_acc, best_dev_epoch, early_count = 0, 0, 0
results = np.zeros([total_epoch, 3, task_num])
for epoch in range(total_epoch):
    print('--- Epoch {}'.format(epoch))
    s_t = time.time()
    model.train()
    # for batch_index in tqdm(range(train_batch)):
    for batch_index in range(train_batch):
#         if batch_index > 2:
#             break
        loss_train = torch.zeros(task_num).cuda()
        for lg_index, lg in enumerate(lang_list):
            inputs = get_data(lg, 'train', dataloader, iter_dataloader)
            outputs = model(inputs, lg_index)
            loss_train[lg_index] = outputs[0]
            results[epoch, 0, lg_index] += outputs[0].item()
                
        weight_update(loss_train, model, optimizer, epoch, batch_index, task_num, clip_grad=True, scheduler=scheduler, avg_cost=results[:,0,:])

    results[epoch, 0, :] /= (batch_index+1)
    print('Train Loss {}'.format(results[epoch,0,:].mean()))
        
    model.eval()
    with torch.no_grad():
        for mode_index, mode in enumerate(['dev', 'test']):
            for lg_index, lg in enumerate(lang_list):
                results[epoch, mode_index+1, lg_index] = get_metric(root_data, model, params.dataset, mode, dataloader, iter_dataloader, lg=lg, lg_index=lg_index)
                
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
    print('Dev Acc/F1 {} avg {}'.format(results[epoch,1,:], results[epoch,1,:].mean()))
    print('Test Acc/F1 {} avg {}'.format(results[epoch,2,:], results[epoch,2,:].mean()))
    print('cost time {}'.format(e_t-s_t))
    
    if results[epoch,1,:].mean() > best_dev_acc:
        best_dev_acc = results[epoch,1,:].mean()
        best_dev_epoch = epoch
    print('Best Dev Epoch {}'.format(best_dev_epoch))
    
    '''
    writer.add_scalar('val/acc_avg', results[epoch,1,:].mean(), epoch)
    writer.add_scalar('test/acc_avg', results[epoch,2,:].mean(), epoch)

    for tn in range(task_num):
        writer.add_scalar('val/{}acc'.format(tn), results[epoch,2,tn], epoch)
        writer.add_scalar('test/{}acc'.format(tn), results[epoch,2,tn], epoch)
    '''