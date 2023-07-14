import os, sys
import numpy as np 

import torch 
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

class FeaturesDataset(Dataset):
    def __init__(self, dataroot, attention_type):
        super(FeaturesDataset, self).__init__()
        self.dataroot = dataroot
        if attention_type == 'backbone_ssa_fc_logit' or attention_type == 'backbone_csa_fc_logit':
            self.features_dir = os.path.join(self.dataroot, 'backbone')
        elif attention_type == 'backbone_fc_ssa_logit' or attention_type == 'backbone_fc_csa_logit':
            self.features_dir = os.path.join(self.dataroot, 'fc_1')
        
        self.features_dir = os.path.join(self.dataroot, 'fc_1')
        self.labels_dir = os.path.join(self.dataroot, 'point_labels')

        self.files = os.listdir(self.features_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        fpath = os.path.join(self.features_dir, self.files[idx])
        labels_path = os.path.join(self.labels_dir, self.files[idx])

        with open(fpath, 'rb') as f:
            feats = np.load(f)
        with open(labels_path, 'rb') as f:
            label = np.load(f)
            label = label.astype(int)
        
        if feats.shape[2] < 10000:
            rem = 10000 - feats.shape[2]
            rep_feats = feats[:, :, :rem, :,].copy()
            rep_labels = label[:rem].copy()

            feats = np.concatenate((feats, rep_feats), axis=2)
            label = np.concatenate((label, rep_labels), axis=0)

        feats = torch.from_numpy(feats)
        label = torch.from_numpy(label)

        return feats, label

class CSADataset(Dataset):
    def __init__(self, all_feats, all_labels, knn_graph, neighbor_feats, K):
        super(CSADataset, self).__init__()
        self.all_feats = all_feats
        self.all_labels = all_labels
        self.knn_graph = knn_graph
        self.neighbor_knn_feats = neighbor_feats
        self.K = K

    def __len__(self):
        return len(self.all_feats)

    def __getitem__(self, idx):
        feats = self.all_feats[idx]
        label = self.all_labels[idx]

        knn_shape_indices = self.knn_graph[idx]
        neighbor_feats = [feats.numpy()] # shape itself in the cluster at idx=0

        for kidx in knn_shape_indices:
            if kidx != idx:
                neighbor_feats.append(self.neighbor_knn_feats[kidx].numpy())
            if len(neighbor_feats) == self.K+1:
                break

        neighbor_feats = torch.from_numpy(np.array(neighbor_feats))

        return feats, label, neighbor_feats

class CSADatasetK(Dataset):
    def __init__(self, dataroot, dataroot_K, knn_graph, K):
        super(CSADatasetK, self).__init__()
        self.dataroot = dataroot
        self.dataroot_K = dataroot_K
        self.K = K 
        self.knn_graph = np.copy(knn_graph)

        self.features_dir = os.path.join(self.dataroot, 'fc_1')
        self.labels_dir = os.path.join(self.dataroot, 'point_labels')
        self.files = os.listdir(self.features_dir)

        self.neighbors_features_dir = os.path.join(self.dataroot_K, 'fc_1')
        self.neighbor_files = os.listdir(self.neighbors_features_dir)

    def __len__(self):
        return len(self.files)

    def get_feats(self, idx):
        fpath = os.path.join(self.neighbors_features_dir, self.neighbor_files[idx])
        with open(fpath, 'rb') as f:
            np_feats = np.load(f)
        if np_feats.shape[2] < 10000:
            rem = 10000 - np_feats.shape[2]
            rep_feats = np_feats[:, :, :rem, :,].copy()
            np_feats = np.concatenate((np_feats, rep_feats), axis=2)
        return np_feats

    def __getitem__(self, idx):
        fpath = os.path.join(self.features_dir, self.files[idx])
        labels_path = os.path.join(self.labels_dir, self.files[idx])
        with open(fpath, 'rb') as f:
            feats = np.load(f)
        with open(labels_path, 'rb') as f:
            label = np.load(f)
            label = label.astype(int)
        
        if feats.shape[2] < 10000:
            rem = 10000 - feats.shape[2]
            rep_feats = feats[:, :, :rem, :,].copy()
            rep_labels = label[:rem].copy()

            feats = np.concatenate((feats, rep_feats), axis=2)
            label = np.concatenate((label, rep_labels), axis=0)

        knn_shape_indices = self.knn_graph[idx]
        neighbor_feats = [np.copy(feats)] # shape itself in the cluster at idx=0
        for kidx in knn_shape_indices:
            if kidx != idx:
                neighbor_np_feats = self.get_feats(kidx)
                neighbor_feats.append(np.copy(neighbor_np_feats))
            if len(neighbor_feats) == self.K+1:
                break

        feats = torch.from_numpy(feats)
        label = torch.from_numpy(label)
        neighbor_feats = torch.from_numpy(np.array(neighbor_feats))

        feats = torch.squeeze(feats, dim=0)
        neighbor_feats = torch.squeeze(neighbor_feats, dim=1)

        return feats, label, neighbor_feats


class FeaturesDatasetforPaper(Dataset):
    def __init__(self, dataroot, attention_type):
        super(FeaturesDatasetforPaper, self).__init__()
        self.dataroot = dataroot
        if attention_type == 'backbone_ssa_fc_logit' or attention_type == 'backbone_csa_fc_logit':
            self.features_dir = os.path.join(self.dataroot, 'backbone')
        elif attention_type == 'backbone_fc_ssa_logit' or attention_type == 'backbone_fc_csa_logit':
            self.features_dir = os.path.join(self.dataroot, 'fc_1')
        
        self.features_dir = os.path.join(self.dataroot, 'fc_1')
        self.labels_dir = os.path.join(self.dataroot, 'point_labels')

        self.files = os.listdir(self.features_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        fpath = os.path.join(self.features_dir, self.files[idx])
        labels_path = os.path.join(self.labels_dir, self.files[idx])

        with open(fpath, 'rb') as f:
            feats = np.load(f)
        with open(labels_path, 'rb') as f:
            label = np.load(f)
            label = label.astype(int)
        
        if feats.shape[2] < 10000:
            rem = 10000 - feats.shape[2]
            rep_feats = feats[:, :, :rem, :,].copy()
            rep_labels = label[:rem].copy()

            feats = np.concatenate((feats, rep_feats), axis=2)
            label = np.concatenate((label, rep_labels), axis=0)

        feats = torch.from_numpy(feats)
        label = torch.from_numpy(label)

        return feats, label, self.files[idx]

class CSADatasetKforPaper(Dataset):
    def __init__(self, dataroot, dataroot_K, knn_graph, K):
        super(CSADatasetKforPaper, self).__init__()
        self.dataroot = dataroot
        self.dataroot_K = dataroot_K
        self.K = K 
        self.knn_graph = np.copy(knn_graph)

        self.features_dir = os.path.join(self.dataroot, 'fc_1')
        self.labels_dir = os.path.join(self.dataroot, 'point_labels')
        self.files = os.listdir(self.features_dir)

        self.neighbors_features_dir = os.path.join(self.dataroot_K, 'fc_1')
        self.neighbor_files = os.listdir(self.neighbors_features_dir)

    def __len__(self):
        return len(self.files)

    def get_feats(self, idx):
        fpath = os.path.join(self.neighbors_features_dir, self.neighbor_files[idx])
        with open(fpath, 'rb') as f:
            np_feats = np.load(f)
        if np_feats.shape[2] < 10000:
            rem = 10000 - np_feats.shape[2]
            rep_feats = np_feats[:, :, :rem, :,].copy()
            np_feats = np.concatenate((np_feats, rep_feats), axis=2)
        return np_feats

    def some_neighbors(self, cat, shape, fpath, knn_shape_indices):
        if cat in fpath and shape in fpath:
            print('fpath:', fpath)
            for kidx in knn_shape_indices:
                npath = os.path.join(self.neighbors_features_dir, self.neighbor_files[kidx])
                print('npath:', npath)
            print()
            sys.exit()

    def __getitem__(self, idx):
        fpath = os.path.join(self.features_dir, self.files[idx])
        labels_path = os.path.join(self.labels_dir, self.files[idx])
        with open(fpath, 'rb') as f:
            feats = np.load(f)
        with open(labels_path, 'rb') as f:
            label = np.load(f)
            label = label.astype(int)
        
        if feats.shape[2] < 10000:
            rem = 10000 - feats.shape[2]
            rep_feats = feats[:, :, :rem, :,].copy()
            rep_labels = label[:rem].copy()

            feats = np.concatenate((feats, rep_feats), axis=2)
            label = np.concatenate((label, rep_labels), axis=0)

        knn_shape_indices = self.knn_graph[idx]
        neighbor_feats = [np.copy(feats)] # shape itself in the cluster at idx=0
        for kidx in knn_shape_indices:
            if kidx != idx:
                neighbor_np_feats = self.get_feats(kidx)
                neighbor_feats.append(np.copy(neighbor_np_feats))
            if len(neighbor_feats) == self.K+1:
                break
        
        self.some_neighbors('Table', 'shape_1531', fpath, knn_shape_indices)
        self.some_neighbors('Lamp', 'shape_245', fpath, knn_shape_indices)
        self.some_neighbors('Chair', 'shape_524', fpath, knn_shape_indices)
        self.some_neighbors('Knife', 'shape_24', fpath, knn_shape_indices)

        feats = torch.from_numpy(feats)
        label = torch.from_numpy(label)
        neighbor_feats = torch.from_numpy(np.array(neighbor_feats))

        feats = torch.squeeze(feats, dim=0)
        neighbor_feats = torch.squeeze(neighbor_feats, dim=1)

        return feats, label, neighbor_feats, self.files[idx]
