from torch.utils.data import DataLoader, Dataset
import os
import torch
import torch.nn.functional as F
import fnmatch
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import Image


class office_Dataset(Dataset):
    def __init__(self, dataset, task, mode):
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                        ])
        if mode != 'trval':
            f = open('./data_txt/{}/{}_{}.txt'.format(dataset, task, mode), 'r')
            self.img_list = f.readlines()
            f.close()
        else:
            f1 = open('./data_txt/{}/{}_train.txt'.format(dataset, task), 'r')
            f2 = open('./data_txt/{}/{}_val.txt'.format(dataset, task), 'r')
            self.img_list = f1.readlines() + f2.readlines()
            f1.close()
            f2.close()
        if dataset == 'office-31':
            self.root_path = '/data/dataset/office31/'
        elif dataset == 'office-home':
            self.root_path = '/data/dataset/office_home/'
        
    def __getitem__(self, i):
        img_path = self.img_list[i][:-1].split(' ')[0]
        y = int(self.img_list[i][:-1].split(' ')[1])
        img = Image.open(self.root_path + img_path).convert('RGB')
        return self.transform(img), y
        
    def __len__(self):
        return len(self.img_list)
    
def office_dataloader(dataset, batchsize):
    if dataset == 'office-31':
        tasks = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-home':
        tasks = ['Art', 'Clipart', 'Product', 'Real_World']
    data_loader = {}
    iter_data_loader = {}
    for k, d in enumerate(tasks):
        data_loader[k] = {}
        iter_data_loader[k] = {}
        for mode in ['train', 'val', 'test', 'trval']:
            shuffle = False if mode == 'test' else True
            drop_last = False if mode == 'test' else True
            txt_dataset = office_Dataset(dataset, d, mode)
            print(d, mode, len(txt_dataset))
            data_loader[k][mode] = DataLoader(txt_dataset, 
                                              num_workers=0, 
                                              pin_memory=True, 
                                              batch_size=batchsize, 
                                              shuffle=shuffle,
                                              drop_last=drop_last)
            iter_data_loader[k][mode] = iter(data_loader[k][mode])
    return data_loader, iter_data_loader
    

def office_dataloader_other(dataset, batchsize):
    if dataset == 'office-31':
        tasks = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-home':
        tasks = ['Art', 'Clipart', 'Product', 'Real_World']
    data_loader = {}
    iter_data_loader = {}
    for k, d in enumerate(tasks):
        data_loader[k] = {}
        iter_data_loader[k] = {}
        for mode in ['train', 'val', 'test', 'trval']:
            shuffle = False if mode == 'test' else True
            # drop_last = False if mode == 'test' else True
            drop_last = True
            txt_dataset = office_Dataset(dataset, d, mode)
            print(d, mode, len(txt_dataset))
            data_loader[k][mode] = DataLoader(txt_dataset, 
                                              num_workers=0, 
                                              pin_memory=True, 
                                              batch_size=batchsize, 
                                              shuffle=shuffle,
                                              drop_last=drop_last)
            iter_data_loader[k][mode] = iter(data_loader[k][mode])
    return data_loader, iter_data_loader