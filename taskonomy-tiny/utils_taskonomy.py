# https://github.com/sunxm2357/AdaShare/blob/master/envs/base_env.py

import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix


def get_seg_loss(seg_pred, seg, seg_num_class, dataroot):
    weight = torch.from_numpy(np.load(os.path.join(dataroot, 'semseg_prior_factor.npy'))).cuda().float()
    loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
    prediction = seg_pred.permute(0, 2, 3, 1).contiguous().view(-1, seg_num_class)
    batch_size = seg_pred.shape[0]
    new_shape = seg_pred.shape[-2:]
    seg_resize = F.interpolate(seg.float(), size=new_shape)
    gt = seg_resize.permute(0, 2, 3, 1).contiguous().view(-1)
    loss = loss_fn(prediction, gt.long())
    return loss

def get_sn_loss(sn_pred, normal, normal_mask=None):
    prediction = sn_pred.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    new_shape = sn_pred.shape[-2:]
    sn_resize = F.interpolate(normal.float(), size=new_shape)
    gt = sn_resize.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    labels = (gt.max(dim=1)[0] < 255)
    if normal_mask is not None:
        normal_mask_resize = F.interpolate(normal_mask.float(), size=new_shape)
        gt_mask = normal_mask_resize.permute(0, 2, 3, 1).contiguous().view(-1, 3)
#         print('/'*10, labels.shape, gt_mask.shape)
        labels = labels*(gt_mask[:,0].int() == 1)

    prediction = prediction[labels]
    gt = gt[labels]

    prediction = F.normalize(prediction)
    gt = F.normalize(gt)
    cosine_similiarity = nn.CosineSimilarity()
    
    return 1 - cosine_similiarity(prediction, gt).mean()


def get_depth_loss(depth_pred, depth, depth_mask=None):
    new_shape = depth_pred.shape[-2:]
    depth_resize = F.interpolate(depth.float(), size=new_shape)
    if depth_mask is not None:
        depth_mask_resize = F.interpolate(depth_mask.float(), size=new_shape)

    binary_mask = (depth_resize != 255) * (depth_mask_resize.int() == 1)

    depth_output = depth_pred.masked_select(binary_mask)
    depth_gt = depth_resize.masked_select(binary_mask)

    l1_loss = nn.L1Loss()
    return l1_loss(depth_output, depth_gt)


def get_keypoint_loss(keypoint_pred, keypoint):
    new_shape = keypoint_pred.shape[-2:]
    keypoint_resize = F.interpolate(keypoint.float(), size=new_shape)
    binary_mask = keypoint_resize != 255

    keypoint_output = keypoint_pred.masked_select(binary_mask)
    keypoint_gt = keypoint_resize.masked_select(binary_mask)
    l1_loss = nn.L1Loss()
    return l1_loss(keypoint_output, keypoint_gt)


def get_edge_loss(edge_pred, edge):
    new_shape = edge_pred.shape[-2:]
    edge_resize = F.interpolate(edge.float(), size=new_shape)
    binary_mask = edge_resize != 255
    edge_output = edge_pred.masked_select(binary_mask)
    edge_gt = edge_resize.masked_select(binary_mask)
    l1_loss = nn.L1Loss()
    return l1_loss(edge_output, edge_gt)


def seg_error(seg_output, seg, seg_num_class, dataroot):
    with torch.no_grad():
        gt = seg.view(-1)
        labels = gt < seg_num_class
        gt = gt[labels].int()

        logits = seg_output.permute(0, 2, 3, 1).contiguous().view(-1, seg_num_class)
        logits = logits[labels]

    #     weight = torch.from_numpy(np.load(os.path.join(dataroot, 'semseg_prior_factor.npy'))).cuda().float()
    #     loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
    #     err = loss_fn(logits, gt.long())

        prediction = torch.argmax(seg_output, dim=1)
        prediction = prediction.unsqueeze(1)

        # pixel acc
        prediction = prediction.view(-1)
        prediction = prediction[labels].int()
        pixelAcc = (gt == prediction).float().mean()

        return prediction.cpu().numpy(), gt.cpu().numpy(), pixelAcc.cpu().numpy()

def normal_error(sn_output, normal, normal_mask=None):
    # normalized, ignored gt and prediction
    with torch.no_grad():
        prediction = sn_output.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        gt = normal.permute(0, 2, 3, 1).contiguous().view(-1, 3)

        labels = gt.max(dim=1)[0] != 255
        if normal_mask is not None:
            gt_mask = normal_mask.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            labels = labels*(gt_mask[:,0].int() == 1)

        gt = gt[labels]
        prediction = prediction[labels]

        gt = F.normalize(gt.float(), dim=1)
        prediction = F.normalize(prediction, dim=1)

    #     cosine_similiarity = nn.CosineSimilarity()
        cos_similarity = F.cosine_similarity(gt, prediction)

        return cos_similarity.cpu().numpy()

def depth_error(depth_output, depth, depth_mask):
    with torch.no_grad():
        if depth_mask is not None:
            binary_mask = (depth != 255) * (depth_mask.int() == 1)

        depth_output_true = depth_output.masked_select(binary_mask)
        depth_gt_true = depth.masked_select(binary_mask)
        abs_err = torch.abs(depth_output_true - depth_gt_true)
        rel_err = torch.abs(depth_output_true - depth_gt_true) / depth_gt_true
        sq_rel_err = torch.pow(depth_output_true - depth_gt_true, 2) / depth_gt_true
        abs_err = torch.sum(abs_err) / torch.nonzero(binary_mask).size(0)
        rel_err = torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)
        sq_rel_err = torch.sum(sq_rel_err) / torch.nonzero(binary_mask).size(0)
        # calcuate the sigma
        term1 = depth_output_true / depth_gt_true
        term2 = depth_gt_true / depth_output_true
        ratio = torch.max(torch.stack([term1, term2], dim=0), dim=0)
        # calcualte rms
        rms = torch.pow(depth_output_true - depth_gt_true, 2)
        rms_log = torch.pow(torch.log10(depth_output_true + 1e-7) - torch.log10(depth_gt_true + 1e-7), 2)

        return abs_err.cpu().numpy(), rel_err.cpu().numpy(), sq_rel_err.cpu().numpy(), ratio[0].cpu().numpy(), \
               rms.cpu().numpy(), rms_log.cpu().numpy()

def keypoint_error(keypoint_output, keypoint):
    with torch.no_grad():
        binary_mask = (keypoint != 255)
        keypoint_output_true = keypoint_output.masked_select(binary_mask)
        keypoint_gt_true = keypoint.masked_select(binary_mask)
        abs_err = torch.abs(keypoint_output_true - keypoint_gt_true).mean()
        return abs_err.cpu().numpy()

def edge_error(edge_output, edge):
    with torch.no_grad():
        binary_mask = (edge != 255)
        edge_output_true = edge_output.masked_select(binary_mask)
        edge_gt_true = edge.masked_select(binary_mask)
        abs_err = torch.abs(edge_output_true - edge_gt_true).mean()
        return abs_err.cpu().numpy()

def compute_loss(pred_dict, gt_dict, dataroot):
    loss_train = torch.zeros(len(pred_dict)).cuda()
    seg_num_class = 17
    for i, (tk, pred) in enumerate(pred_dict.items()):
        if tk == 'seg':
            loss_train[i] = get_seg_loss(pred, gt_dict['seg'], seg_num_class, dataroot)
        if tk == 'depth':
            loss_train[i] = get_depth_loss(pred, gt_dict['depth'], gt_dict['depth_mask'])
        if tk == 'sn':
            loss_train[i] = get_sn_loss(pred, gt_dict['normal'], gt_dict['normal_mask'])
        if tk == 'keypoint':
            loss_train[i] = get_keypoint_loss(pred, gt_dict['keypoint'])
        if tk == 'edge':
            loss_train[i] = get_edge_loss(pred, gt_dict['edge'])
    return loss_train
            

class PerformanceMeter(object):
    def __init__(self, tasks, dataroot):
        self.batch_size = []
        self.records = {}
        self.val_metrics = {}
        self.tasks = tasks
        self.dataroot = dataroot
        self.num_seg_cls = 17
        if 'seg' in self.tasks:
            self.records['seg'] = {'mIoUs': [], 'pixelAccs': [],  'errs': [], 
                                   'conf_mat': np.zeros((self.num_seg_cls, self.num_seg_cls)),
                                   'labels': np.arange(self.num_seg_cls)}
        if 'sn' in self.tasks:
            self.records['sn'] = {'cos_similaritys': []}
        if 'depth' in self.tasks:
            self.records['depth'] = {'abs_errs': [], 'rel_errs': [], 'sq_rel_errs': [], 'ratios': [], 'rms': [], 'rms_log': []}
        if 'keypoint' in self.tasks:
            self.records['keypoint'] = {'errs': []}
        if 'edge' in self.tasks:
            self.records['edge'] = {'errs': []}
            
    def update(self, pred_dict, gt_dict):
        if 'seg' in self.tasks:
            prediction, gt, pixelAcc = seg_error(pred_dict['seg'], gt_dict['seg'], self.num_seg_cls, self.dataroot)
            new_mat = confusion_matrix(gt, prediction, labels=self.records['seg']['labels'])
            assert (self.records['seg']['conf_mat'].shape == new_mat.shape)
            self.records['seg']['conf_mat'] += new_mat
            self.records['seg']['pixelAccs'].append(pixelAcc)
#             self.records['seg']['errs'].append(err)
        if 'sn' in self.tasks:
            cos_similarity = normal_error(pred_dict['sn'], gt_dict['normal'], gt_dict['normal_mask'])
            self.records['sn']['cos_similaritys'].append(cos_similarity)
        if 'depth' in self.tasks:
            abs_err, rel_err, sq_rel_err, ratio, rms, rms_log = depth_error(pred_dict['depth'], gt_dict['depth'], gt_dict['depth_mask'])
            self.records['depth']['abs_errs'].append(abs_err)
            self.records['depth']['rel_errs'].append(rel_err)
#             self.records['depth']['sq_rel_errs'].append(sq_rel_err)
#             self.records['depth']['ratios'].append(ratio)
#             self.records['depth']['rms'].append(rms)
#             self.records['depth']['rms_log'].append(rms_log)
        if 'keypoint' in self.tasks:
            err = keypoint_error(pred_dict['keypoint'], gt_dict['keypoint'])
            self.records['keypoint']['errs'].append(err)
        if 'edge' in self.tasks:
            err = edge_error(pred_dict['edge'], gt_dict['edge'])
            self.records['edge']['errs'].append(err)
        self.batch_size.append(list(pred_dict.values())[0].shape[0])
        
    def get_score(self):
        if 'seg' in self.tasks:
            self.val_metrics['seg'] = {}
            jaccard_perclass = []
            for i in range(self.num_seg_cls):
                if not self.records['seg']['conf_mat'][i, i] == 0:
                    jaccard_perclass.append(self.records['seg']['conf_mat'][i, i] / 
                                            (np.sum(self.records['seg']['conf_mat'][i, :]) +
                                            np.sum(self.records['seg']['conf_mat'][:, i]) -
                                            self.records['seg']['conf_mat'][i, i]))

            self.val_metrics['seg']['mIoU'] = np.sum(jaccard_perclass) / len(jaccard_perclass)

            self.val_metrics['seg']['Pixel Acc'] = (np.array(self.records['seg']['pixelAccs']) * np.array(self.batch_size)).sum() / sum(self.batch_size)

#             self.val_metrics['seg']['err'] = (np.array(self.records['seg']['errs']) * np.array(self.batch_size)).sum() / sum(self.batch_size)

        if 'sn' in self.tasks:
            self.val_metrics['sn'] = {}
            overall_cos = np.clip(np.concatenate(self.records['sn']['cos_similaritys']), -1, 1)

            angles = np.arccos(overall_cos) / np.pi * 180.0
#             self.val_metrics['sn']['cosine_similarity'] = overall_cos.mean()
            self.val_metrics['sn']['Angle Mean'] = np.mean(angles)
            self.val_metrics['sn']['Angle Median'] = np.median(angles)
#             self.val_metrics['sn']['Angle RMSE'] = np.sqrt(np.mean(angles ** 2))
            self.val_metrics['sn']['Angle 11.25'] = np.mean(np.less_equal(angles, 11.25)) * 100
            self.val_metrics['sn']['Angle 22.5'] = np.mean(np.less_equal(angles, 22.5)) * 100
            self.val_metrics['sn']['Angle 30'] = np.mean(np.less_equal(angles, 30.0)) * 100
#             self.val_metrics['sn']['Angle 45'] = np.mean(np.less_equal(angles, 45.0)) * 100

        if 'depth' in self.tasks:
            self.val_metrics['depth'] = {}
            self.records['depth']['abs_errs'] = np.stack(self.records['depth']['abs_errs'], axis=0)
            self.records['depth']['rel_errs'] = np.stack(self.records['depth']['rel_errs'], axis=0)
#             self.records['depth']['sq_rel_errs'] = np.stack(self.records['depth']['sq_rel_errs'], axis=0)
#             self.records['depth']['ratios'] = np.concatenate(self.records['depth']['ratios'], axis=0)
#             self.records['depth']['rms'] = np.concatenate(self.records['depth']['rms'], axis=0)
#             self.records['depth']['rms_log'] = np.concatenate(self.records['depth']['rms_log'], axis=0)
#             self.records['depth']['rms_log'] = self.records['depth']['rms_log'][~np.isnan(self.records['depth']['rms_log'])]
            self.val_metrics['depth']['abs_err'] = (self.records['depth']['abs_errs'] * np.array(self.batch_size)).sum() / sum(self.batch_size)
            self.val_metrics['depth']['rel_err'] = (self.records['depth']['rel_errs'] * np.array(self.batch_size)).sum() / sum(self.batch_size)
#             self.val_metrics['depth']['sq_rel_err'] = (self.records['depth']['sq_rel_errs'] * np.array(self.batch_size)).sum() / sum(
#                 self.batch_size)
#             self.val_metrics['depth']['sigma_1.25'] = np.mean(np.less_equal(self.records['depth']['ratios'], 1.25)) * 100
#             self.val_metrics['depth']['sigma_1.25^2'] = np.mean(np.less_equal(self.records['depth']['ratios'], 1.25 ** 2)) * 100
#             self.val_metrics['depth']['sigma_1.25^3'] = np.mean(np.less_equal(self.records['depth']['ratios'], 1.25 ** 3)) * 100
#             self.val_metrics['depth']['rms'] = (np.sum(self.records['depth']['rms']) / len(self.records['depth']['rms'])) ** 0.5
            # val_metrics['depth']['rms_log'] = (np.sum(records['depth']['rms_log']) / len(records['depth']['rms_log'])) ** 0.5

        if 'keypoint' in self.tasks:
            self.val_metrics['keypoint'] = {}
            self.val_metrics['keypoint']['err'] = (np.array(self.records['keypoint']['errs']) * np.array(self.batch_size)).sum() / sum(
                self.batch_size)

        if 'edge' in self.tasks:
            self.val_metrics['edge'] = {}
            self.val_metrics['edge']['err'] = (np.array(self.records['edge']['errs']) * np.array(self.batch_size)).sum() / sum(
                self.batch_size)

        return self.val_metrics
