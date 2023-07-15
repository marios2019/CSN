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
from tqdm import tqdm 

from features_data_loader import FeaturesDataset, CSADataset, CSADatasetK
from csa_models import * 
from utils import load_trained_ssa_layers

parser = argparse.ArgumentParser()
parser.add_argument('--jobid', type=int, default=0)
parser.add_argument('--logs_dir', type=str, default='logs/csa_n_heads_1_K_1/Bed')
parser.add_argument('--ssa_logs_dir', type=str, default='logs/ssa_n_heads_1/Bed')

parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--testing', action='store_true')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--partname', type=str, default='Bed')
parser.add_argument('--train_iters', type=int, default=3000)
parser.add_argument('--num_classes', type=int, default=15)
parser.add_argument('--attention_type', type=str, default='csa')
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--n_heads', type=int, default=1)
parser.add_argument('--gradient_accumulation_steps', type=int, default=2)

big_classes = ['Chair', 'Lamp', 'StorageFurniture', 'Table']

args = parser.parse_args() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_after = 200 
if args.partname in big_classes:
    eval_after = 1000

def gpu_mem():
    t = torch.cuda.get_device_properties(0).total_memory * 1e-9
    r = torch.cuda.memory_reserved(0) * 1e-9
    a = torch.cuda.memory_allocated(0) * 1e-9
    # f = r-a  # free inside reserved

    ret = f't: {t:.2f}GB a: {a:.2f}GB r: {r:.2f}GB'
    return ret

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
    print('model loaded from', path)
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

def update_knn_graphs(train_root, test_root, model, train_loader, test_loader, K, logs_dir):
    
    if args.partname in big_classes:
        candidate_shape_indices = model.get_center_shape_indices(train_loader) 

        train_knn_cand_shape_indices = model.get_knn_graph_big(train_loader, train_loader, candidate_shape_indices, args.K).cpu().numpy()
        train_knn_graph = []
        for i in range(len(train_knn_cand_shape_indices)):
            knn = []
            for j in range(len(train_knn_cand_shape_indices[i])):
                knn += [candidate_shape_indices[train_knn_cand_shape_indices[i][j]]]
            train_knn_graph += [knn.copy()]

        test_knn_cand_shape_indices = model.get_knn_graph_big(test_loader, train_loader, candidate_shape_indices, args.K).cpu().numpy()
        test_knn_graph = []
        for i in range(len(test_knn_cand_shape_indices)):
            knn = []
            for j in range(len(test_knn_cand_shape_indices[i])):
                knn += [candidate_shape_indices[test_knn_cand_shape_indices[i][j]]]
            test_knn_graph += [knn.copy()]
    else:
        train_ssa_feats = model.get_all_feats(logs_dir, train_loader, K, 'train')
        test_ssa_feats = model.get_all_feats(logs_dir, test_loader, K, 'test')

        train_ssa_feats = train_ssa_feats.to(device)
        test_ssa_feats = test_ssa_feats.to(device)
        train_knn_graph = model.get_knn_graph(train_ssa_feats, train_ssa_feats, K).cpu().numpy()
        test_knn_graph = model.get_knn_graph(test_ssa_feats, train_ssa_feats, K).cpu().numpy()

        del train_ssa_feats
        del test_ssa_feats
        torch.cuda.empty_cache()


    csa_dataset = CSADatasetK(train_root, train_root, train_knn_graph, K)
    csa_train_dataloader = DataLoader(csa_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    csa_dataset = CSADatasetK(test_root, train_root, test_knn_graph, K)
    csa_test_dataloader = DataLoader(csa_dataset, 1, shuffle=False, num_workers=args.num_workers)

    return csa_train_dataloader, csa_test_dataloader


def update_best_model_metrics(best_IoU, val_IoU, model, save_name, logs_fp):
    if val_IoU > best_IoU:
        best_IoU = val_IoU 
        torch.save(model.state_dict(), save_name)
        logprint(f'model saved to: {save_name}!', logs_fp)

        df = pd.DataFrame([[val_IoU*100]], columns=columns)
        df.to_csv(df_path)
        logprint(f'val_IoU {val_IoU*100} saved to {df_path}', logs_fp)
    
    return best_IoU

def train_layers(model, train_dataloader, val_dataloader, best_IoU, optimizer, scheduler, num_class, weight_decay, device, save_name, logs_fp):
    model.train()
    running_loss = 0.

    for i, data in enumerate(train_dataloader):

        optimizer.zero_grad()
        feats, label, neighbor_feats = data  # neighbor contains the self shape at idx=0, dim=1
        feats = feats.to(device)
        label = label.to(device)

        output = model(feats, 'test', neighbor_feats)
        loss, accu = loss_functions_seg(output, label, num_class, weight_decay, mask=0)
        loss /= args.gradient_accumulation_steps

        if torch.isnan(loss):
            loss *= 0.
        else:
            running_loss += loss.item() 
    
        loss.backward()

        
        if (i+1) % args.gradient_accumulation_steps == 0 or (i+1) == len(train_dataloader):
            optimizer.step()
            optimizer.zero_grad()
    
        if args.testing:
            break 
        
    running_loss /= len(train_dataloader)
    return running_loss, best_IoU

def validate_layers(model, val_dataloader, class_num, weight_decay, device):
    model.eval()
    
    val_intsc = [0] * class_num
    val_union = [0] * class_num
    running_loss = 0.
    for i, data in enumerate(val_dataloader):
        feats, label, neighbor_feats = data  # neighbor contains the self shape at idx=0, dim=1
        feats = feats.to(device)
        label = label.to(device)
        neighbor_feats = neighbor_feats.to(device)

        output = model(feats, 'test', neighbor_feats)
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

train_dataset = FeaturesDataset(dataroot.format('train', args.partname), args.attention_type)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

test_dataset = FeaturesDataset(dataroot.format('test', args.partname), args.attention_type)
test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=args.num_workers)

model = get_model(args.attention_type, args.num_classes, args.n_heads, args.K).to(device)
model = load_trained_ssa_layers(model, args.ssa_logs_dir)
logprint('trained_ssa_layers imported!', logs_fp)
model.device = device
save_name = os.path.join(logs_dir, 'trained_layers.pth')

knn_graph_dir = f'logs/knn_graphs/n_heads_{args.n_heads}/{args.partname}'
with open(os.path.join(knn_graph_dir, 'train.npy'), 'rb') as f:
    train_knn_graph = np.load(f)
with open(os.path.join(knn_graph_dir, 'test.npy'), 'rb') as f:
    test_knn_graph = np.load(f)

print('train graph:', train_knn_graph.shape)
print('test graph:', test_knn_graph.shape)

csa_dataset = CSADatasetK(train_root, train_root, train_knn_graph, args.K)
csa_train_dataloader = DataLoader(csa_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

csa_dataset = CSADatasetK(test_root, train_root, test_knn_graph, args.K)
csa_test_dataloader = DataLoader(csa_dataset, 1, shuffle=False, num_workers=args.num_workers)
torch.cuda.empty_cache()


epochs = 24 
T = epochs
running_lr = args.lr

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=running_lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) # change to cosine decay

best_IoU = 0.
columns = [args.partname]
df_path = os.path.join(args.logs_dir, 'test_summaries.csv')

for t in range(T):
    logprint('iter {}/{}'.format(t+1, T), logs_fp)
    logprint(f'curr lr: {scheduler.get_last_lr()[0]}', logs_fp)

    train_loss, best_IoU = train_layers(model, csa_train_dataloader, csa_test_dataloader, best_IoU, optimizer, scheduler, args.num_classes, args.weight_decay, device, save_name, logs_fp)
    
    if (t+1) % 1 == 0 or args.testing:
        val_IoU, val_loss = validate_layers(model, csa_test_dataloader, args.num_classes, args.weight_decay, device)
        logprint(f'iter: {t+1}/{T} train_loss: {train_loss} val_loss: {val_loss} val_IoU: {val_IoU*100} best_IoU: {best_IoU}', logs_fp)

        if val_IoU > best_IoU:
            best_IoU = val_IoU 
            torch.save(model.state_dict(), save_name)
            logprint(f'model saved to: {save_name}!', logs_fp)

            df = pd.DataFrame([[val_IoU*100]], columns=columns)
            df.to_csv(df_path)
            logprint(f'test IoU saved to {df_path}', logs_fp)

        logprint('-'*100, logs_fp)
    
    if (t+1) == 10 or (t+1) == (3*T) // 4:
        scheduler.step()

    if args.testing:
        break   

model.load_state_dict(torch.load(save_name))
logprint(f'best model loaded from {save_name}', logs_fp)

logprint(f'Updating KNN graph....', logs_fp)
csa_train_dataloader, csa_test_dataloader = update_knn_graphs(train_root, test_root, model, train_loader, test_loader, args.K, args.logs_dir)
logprint(f'KNN graph UPDATED!', logs_fp)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) # change to cosine decay

for t in range(T):
    logprint('iter {}/{}'.format(t+1, T), logs_fp)
    logprint(f'curr lr: {scheduler.get_last_lr()[0]}', logs_fp)

    train_loss, best_IoU = train_layers(model, csa_train_dataloader, csa_test_dataloader, best_IoU, optimizer, scheduler, args.num_classes, args.weight_decay, device, save_name, logs_fp)
    
    if (t+1) % 1 == 0 or args.testing:
        val_IoU, val_loss = validate_layers(model, csa_test_dataloader, args.num_classes, args.weight_decay, device)
        logprint(f'iter: {t+1}/{T} train_loss: {train_loss} val_loss: {val_loss} val_IoU: {val_IoU} best_IoU: {best_IoU}', logs_fp)

        if val_IoU > best_IoU:
            best_IoU = val_IoU 
            torch.save(model.state_dict(), save_name)
            logprint(f'model saved to: {save_name}!', logs_fp)

            df = pd.DataFrame([[val_IoU*100]], columns=columns)
            df.to_csv(df_path)
            logprint(f'test IoU saved to {df_path}', logs_fp)

        logprint('-'*100, logs_fp)
    
    if (t+1) == 10 or (t+1) == (3*T) // 4:
        scheduler.step()
        
    if args.testing:
        break   

logprint('---- Validation ----', logs_fp)
model.load_state_dict(torch.load(save_name))
logprint(f'best model loaded from {save_name}', logs_fp)

val_IoU, val_loss = validate_layers(model, csa_test_dataloader, args.num_classes, args.weight_decay, device)
logprint(f'Final val_IoU: {val_IoU*100}', logs_fp)

df = pd.DataFrame([[val_IoU*100]], columns=columns)
df.to_csv(df_path)
logprint(f'test IoU saved to {df_path}', logs_fp)