import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision, torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os, sys
import pandas as pd
import time
import types 
import argparse 
import numpy as np

from features_data_loader import FeaturesDataset, CSADataset, CSADatasetKforPaper
from csa_models import * 
from utils import load_trained_ssa_layers


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--jobid', type=int, default=0)
parser.add_argument('--logs_dir', type=str, default='logs/pretrained_models/run_1/csa_n_heads_8_K_4/Bed')
parser.add_argument('--knn_graphs', type=str, default='logs/pretrained_models/knn_graphs')

parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--testing', action='store_true')
parser.add_argument('--batch_size', type=int, default=1)

parser.add_argument('--partname', type=str, default='Bed')
parser.add_argument('--num_classes', type=int, default=15)
parser.add_argument('--attention_type', type=str, default='csa')
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--n_heads', type=int, default=8)

args = parser.parse_args() 

def createdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created!')

def logprint(log, logs_fp=None):
    if not isinstance(logs_fp, type(None)):
        with open(logs_fp, 'a') as f:
            f.write(log + '\n')
    print(log)

def torch_save(obj, name, path):
    torch.save({name:obj}, path)
    print(f'{path} saved!')

def torch_load(name, path):
    ckpt = torch.load(path)
    return ckpt[name]

def label_accuracy(label, label_gt):
    label_gt = label_gt.to(torch.int64)
    accuracy = torch.mean((label == label_gt).float())
    return accuracy

def softmax_accuracy(logit, label):
    predict = torch.argmax(logit, dim=1)
    accu = label_accuracy(predict, label.int())
    return accu

def softmax_loss(logit, label_gt, num_class, label_smoothing=0.0):

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    loss = criterion(logit, label_gt)
    return loss

def loss_functions_seg(logit, label_gt, num_class, weight_decay, mask=0):
    label_gt = label_gt.view(-1)

    logit = torch.squeeze(logit, dim=-1)
    logit = torch.permute(logit, (0,2,1))
    logit = logit.contiguous().view(-1, args.num_classes)
    
    label_mask = label_gt > mask  # filter label 0 (for partnet)
    indices = torch.where(label_mask)
    masked_logit = logit[indices[0]]
    masked_label = label_gt[indices[0]]

    loss = softmax_loss(masked_logit, masked_label, num_class)
    accu = softmax_accuracy(masked_logit, masked_label)
    return loss, accu

def IoU_per_shape(pred, label, class_num, mask=0):
    # Set mask to 0 to filter unlabeled points, whose label is 0
    pred = torch.squeeze(pred, dim=-1)
    pred = torch.permute(pred, (0,2,1))
    pred = pred.contiguous().view(-1, args.num_classes)
    pred = torch.argmax(pred, dim=1)

    label = label.view(-1)

    label_mask = label > mask  # mask out label
    indices = torch.where(label_mask)

    masked_pred = pred[indices[0]]
    masked_label = label[indices[0]]

    intsc, union, ious = [None] * class_num, [None] * class_num, [None] * class_num
    iou_avg = 0.
    for k in range(class_num):
        pk, lk = masked_pred == k, masked_label == k
        intsc[k] = torch.sum((pk & lk).float())
        union[k] = torch.sum((pk | lk).float())
        ious[k] = intsc[k] / (union[k] + 1.e-10)
        iou_avg = iou_avg + ious[k]
    iou_avg /= (class_num - 1)
    return intsc, union

def validate_layers(model, val_dataloader, class_num, weight_decay, device, pred_root):
    model.eval()
    
    running_loss = 0.
    running_shape_iou = 0. 
    total_shapes = 0.
    
    val_intsc = [0] * class_num
    val_union = [0] * class_num

    for i, data in enumerate(val_dataloader):
        
        feats, label, neighbor_feats, file_name = data  # neighbor contains the self shape at idx=0, dim=1
        feats = feats.to(device)
        label = label.to(device)
        neighbor_feats = neighbor_feats.to(device)

        output = model(feats, 'test', neighbor_feats)
        pred = torch.squeeze(output)
        pred = torch.permute(pred, (1,0))
        pred = torch.argmax(pred, dim=-1).detach().cpu().numpy()

        loss, accu = loss_functions_seg(output, label, class_num, weight_decay, mask=0)
        
        if torch.isnan(loss): 
            continue
        running_loss += loss.item()

        batch_intsc, batch_union = IoU_per_shape(output, label, class_num)
        
        for k in range(class_num):
            val_intsc[k] += batch_intsc[k].item()
            val_union[k] += batch_union[k].item()
        
        if args.testing:
            break 

    iou_avg = 0.
    for k in range(class_num):
        iou_avg = iou_avg + val_intsc[k] / (val_union[k] + 1.e-10)
    iou_avg /= (class_num-1)

    running_loss /= len(val_dataloader)

    return iou_avg, running_loss

logs_dir = args.logs_dir
logs_fp = os.path.join(logs_dir, 'logs.txt')
createdirs(logs_dir)

partname = args.partname

logprint(f'device:{device}')

dataroot = '/work/siddhantgarg_umass_edu/int-vis/O-CNN/tensorflow/script/logs/partnet_weights_and_logs/partnet_finetune/{}_data_features/{}' # change to your data root

train_root = dataroot.format('train', args.partname)
test_root = dataroot.format('test', args.partname)

model = get_model(args.attention_type, args.num_classes, args.n_heads, args.K).to(device)
model.device = device
model_path = os.path.join(logs_dir, 'trained_layers.pth')
model.load_state_dict(torch.load(model_path))
logprint(f'model wts loaded from {model_path}!', logs_fp)

knn_graph_dir = f'logs/knn_graphs/n_heads_{args.n_heads}/{args.partname}'
with open(os.path.join(knn_graph_dir, 'test.npy'), 'rb') as f:
    test_knn_graph = np.load(f)

print('test graph:', test_knn_graph.shape)

csa_test_dataloader = DataLoader(csa_dataset, 1, shuffle=False, num_workers=args.num_workers)
torch.cuda.empty_cache()

logprint('---- Validation ----', logs_fp)

val_IoU, val_loss = validate_layers(model, csa_test_dataloader, args.num_classes, args.weight_decay, device)
logprint(f'Final shape_IoU: {val_IoU*100}', logs_fp)

columns = [args.partname]
df_path = os.path.join(args.logs_dir, 'part_IoU_summaries.csv')

df = pd.DataFrame([[val_IoU*100]], columns=columns)
df.to_csv(df_path)
logprint(f'test IoU saved to {df_path}', logs_fp)
