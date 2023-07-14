import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys, math
import numpy as np
from sklearn.cluster import KMeans
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gpu_mem():
    t = torch.cuda.get_device_properties(0).total_memory * 1e-9
    r = torch.cuda.memory_reserved(0) * 1e-9
    a = torch.cuda.memory_allocated(0) * 1e-9
    ret = f't: {t:.2f}GB a: {a:.2f}GB r: {r:.2f}GB'
    return ret

def torch_save(obj, name, path):
    torch.save({name:obj}, path)
    print(f'{path} saved!')

def torch_load(name, path):
    ckpt = torch.load(path)
    ret = ckpt[name]
    print(f'loaded {name} from {path}')
    return ret

def save_npy(arr, path):
    with open(path, 'wb') as f:
        np.save(f, arr)
        print(f'saved to {path}')

def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created!')

class MultiHeadAttention(nn.Module):
    """
        Multi-Head Attention module
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def self_attention(self, x):
        x = torch.permute(torch.squeeze(x, dim=-1), (0, 2, 1))

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = x.size(0), x.size(1), x.size(1), x.size(1)

        residual = x

        q = self.w_qs(x).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.norm(q)
        return q, attn

    def forward(self, Q, K, V, mode):

        iters = 20
        mini_bs = 500
        ret = None 
        for i in range(iters):
            indices = torch.arange(i*500, (i+1)*500)
            q = Q[:, :, indices, :]
            k = K[:, :, indices, :]
            v = V[:, :, indices, :]

            q = torch.permute(torch.squeeze(q, dim=-1), (0, 2, 1))
            k = torch.permute(torch.squeeze(k, dim=-1), (0, 2, 1))
            v = torch.permute(torch.squeeze(v, dim=-1), (0, 2, 1))

            d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
            sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

            residual = q

            # Pass through the pre-attention projection: b x lq x (n*dv)
            # Separate different heads: b x lq x n x dv
            q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
            k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

            # Transpose for attention dot product: b x n x lq x dv
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

            q, attn = self.attention(q, k, v)

            # Transpose to move the head dimension back: b x lq x n x dv
            # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
            q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
            q = self.dropout(self.fc(q))
            q += residual

            q = self.norm(q)

            if isinstance(ret, type(None)):
                ret = q
            else:
                ret = torch.cat((ret, q), dim=1)

        return ret, attn


class ScaledDotProductAttention(nn.Module):
  """
    Scaled Dot-Product Attention
  """

  def __init__(self, temperature, attn_dropout=0.1):
    super().__init__()
    self.temperature = temperature
    self.dropout = nn.Dropout(attn_dropout)

  def forward(self, q, k, v):
    attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

    attn = self.dropout(F.softmax(attn, dim=-1))
    output = torch.matmul(attn, v)

    return output, attn

class CrossShapeAt(nn.Module):
    def __init__(self, num_classes, d_model, n_heads, K=None, d_k=256, d_v=256, attention_type='ssa', after_fc=False, device=None):
        super(CrossShapeAt, self).__init__()

        self.fc_1 = self.octree_conv1x1_bn_relu(928, 256)
        self.logit = self.octree_conv1x1(256, num_classes)
        self.attention = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        
        self.attention_type = attention_type
        self.after_fc = after_fc
        self.device = device

        if 'csa' in self.attention_type:
            self.K = K
            self.compatibility_q = nn.Linear(256, 256)
            self.compatibility_k = nn.Linear(256, 256)
    
    def octree_conv1x1_bn_relu(self, nin, nout):
        layer = nn.Sequential(
            self.octree_conv1x1_bn(nin, nout),
            nn.ReLU()
        ) 
        return layer

    def octree_conv1x1_bn(self, nin, nout):
        layer = nn.Sequential(
            self.octree_conv1x1(nin, nout, use_bias=False),
            nn.BatchNorm2d(nout)
        ) 
        return layer

    def octree_conv1x1(self, nin, nout, use_bias=False):
        layer = nn.Conv2d(nin, nout, kernel_size=1, stride=1, padding='same', bias=use_bias)
        nn.init.xavier_uniform_(layer.weight)
        return layer
    
    def forward(self, x, mode, neighbor_feats=None):
        if self.attention_type == 'ssa':
            x = self.forward_ssa(x, mode)
        if self.attention_type == 'csa':
            x = self.forward_csa(x, neighbor_feats, mode)
        if self.attention_type == 'fcf_csaf_logitf':
            x = self.forward_fcf_csaf_logitf(x, neighbor_feats, mode)
        return x

    def forward_ssa(self, x, mode):   
        if self.after_fc:
            x, self_att = self.get_ssa_feats(x, mode) # (1, 256, H, 1)
        x = self.logit(x) # (1, 256, H, 1)
        return x
    
    def forward_csa(self, x, x_neighbors, mode):
        if self.after_fc:
            x = self.get_csa_feats(x, x_neighbors, mode) # (1, 256, H, 1)
        
        x = self.logit(x) # (1, 256, H, 1)
        return x

    def get_ssa_feats(self, x, mode):
        x, self_att = self.attention(x, x, x,  mode)  # x: (B, H, 256)
        x = torch.unsqueeze(torch.permute(x, (0, 2, 1)), dim=-1) # (B, 256, H, 1)
        return x, self_att

    def get_csa_feats(self, x, x_neighbors, mode):
        y_q, _ = self.get_ssa_feats(x, mode) # ssa_feats : (B, 256, H, 1)
        y_q = y_q.squeeze(dim=-1).permute(0,2,1) # (B, 10000, 256)
        y_q = y_q.mean(dim=1) # (B, 256)
        y_k = y_q
        for k in range(1, x_neighbors.shape[1]):
            x_k = x_neighbors[:, k, :, :, :]
            x_k = x_k.to(device) 
            n_x_ssa, _ = self.get_ssa_feats(x_k, mode)
            n_x_ssa = n_x_ssa.squeeze(dim=-1).permute(0,2,1) # (B, 10000, 256)
            n_x_ssa = n_x_ssa.mean(dim=1) # (B, 256)
            y_k = torch.cat((y_k, n_x_ssa), dim=0) # (B*K, 256)

        u_q = self.compatibility_q(y_q)
        u_q = F.normalize(u_q, dim=-1) 

        u_k = self.compatibility_k(y_k)
        u_k = F.normalize(u_k, dim=-1)
        u_k = u_k.view(x.shape[0], -1, u_k.shape[1]) # (B, K, 256)

        compatibility = torch.matmul(u_q.unsqueeze(1), u_k.permute(0,2,1)).squeeze(1) # (B, 1, K).squeeze(1)-> (B, K)
        compatibility = F.softmax(compatibility, dim=-1)

        csa_feats, _ = self.attention(x, x, x, mode) # ssa_feats : (B, H, 256)
        csa_feats = compatibility[:, 0].unsqueeze(-1).unsqueeze(-1) * csa_feats
        for k in range(1, x_neighbors.shape[1]):  
            x_k = x_neighbors[:, k, :, :, :] 
            x_k = x_k.to(device)
            csa, _ = self.attention(x, x_k, x_k, mode)
            csa_feats += compatibility[:, k].unsqueeze(-1).unsqueeze(-1) * csa # (B, H, 256)
        
        csa_feats = torch.unsqueeze(torch.permute(csa_feats, (0, 2, 1)), dim=-1) # (B, 256, H, 1)
        
        return csa_feats

    def get_retrieval_measure(self, ssa_feats_1, ssa_feats_2):
        '''
            (ssa_feats_1, ssa_feats_2): ssa_feats of test shape, train shape (while testing)
                                        or both ssa_feats of train shape (while training)
        '''

        ret_measure = None
        for i in range(ssa_feats_1.shape[0]):
            f1_ret = None
            f1 = F.normalize(ssa_feats_1[i], dim=-1)
            for f2 in ssa_feats_2:
                f2 = F.normalize(f2, dim=-1)
                cos_score = torch.matmul(f1.unsqueeze(0).unsqueeze(1), f2.unsqueeze(0).permute(0,2,1))
                r = cos_score.max(-1)[0].mean(-1)
                if isinstance(f1_ret, type(None)):
                    f1_ret = r.clone()
                else:
                    f1_ret = torch.cat((f1_ret, r), dim=1)
            if isinstance(ret_measure, type(None)):
                ret_measure = f1_ret.clone()
            else:
                ret_measure = torch.cat((ret_measure, f1_ret.clone()), dim=0)
        
        return ret_measure


    def get_knn_graph(self, ssa_feats_1, ssa_feats_2, K):
        
        '''
            (ssa_feats_1, ssa_feats_2): ssa_feats of test shape, train shape (while testing)
                                        or both ssa_feats of train shape (while training)
        '''
        
        retrieval_measure = self.get_retrieval_measure(ssa_feats_1, ssa_feats_2)
        scores, knn_graph = retrieval_measure.topk(K+1, -1) # K+1: includes other shape and itself.
        retrieval_measure = None
        return knn_graph

    def get_all_feats(self, logs_dir, train_dataloader, K, mode):
        ssa_feats = None 

        for i, data in enumerate(train_dataloader):
            feats, label = data 
            feats = feats.to(device)
            label = label.to(device)
            feats = torch.squeeze(feats, dim=1) # feats: (B, 256, H, 1)
            
            with torch.no_grad():
                batch_ssa_feats, _ = self.get_ssa_feats(feats, mode) # batch_ssa_feats: (B, 256, H, 1)

            if isinstance(ssa_feats, type(None)):
                ssa_feats = batch_ssa_feats.detach().cpu()
            else:
                ssa_feats = torch.concat((ssa_feats, batch_ssa_feats.detach().cpu()), dim=0) # ssa_feats: (N, 256, H, 1)

        ssa_feats = torch.permute(torch.squeeze(ssa_feats, dim=-1), (0,2,1)) # ssa_feats: (N, H, 256)
        return ssa_feats

    def get_center_shape_indices(self, train_loader):

        ssa_global_feats = None 

        for i, data in enumerate(train_loader):
            feats, label = data 
            feats = feats.to(device)
            feats = torch.squeeze(feats, dim=1) # feats: (B, 256, H, 1)
            
            batch_ssa_feats, _ = self.get_ssa_feats(feats, 'test') # batch_ssa_feats: (B, 256, H, 1)
            batch_ssa_feats = torch.permute(torch.squeeze(batch_ssa_feats, dim=-1), (0,2,1)) # batch_ssa_feats: (B, H, 256)
            batch_ssa_feats = torch.amax(batch_ssa_feats, dim=1) # batch_ssa_feats: (B, 256)

            if isinstance(ssa_global_feats, type(None)):
                ssa_global_feats = batch_ssa_feats.detach().clone()
            else:
                ssa_global_feats = torch.concat((ssa_global_feats, batch_ssa_feats.detach().clone()), dim=0) # ssa_feats: (N, 256)
        ssa_global_feats = ssa_global_feats.cpu().numpy()

        n_centers = len(ssa_global_feats) // 10

        kmeans = KMeans(n_clusters=n_centers, random_state=0, n_init=10).fit(ssa_global_feats)
        cluster_labels = kmeans.labels_ 
        final_centers = kmeans.cluster_centers_

        final_centers = np.expand_dims(final_centers, axis=1)
        dists = (final_centers - ssa_global_feats)**2 
        dists = np.sum(dists, axis=-1)
        center_indices = np.argmin(dists, axis=-1)

        return center_indices

    def get_candidate_ssa_feats(self, data_loader, candidate_shape_indices):
        ssa_feats = None 

        counter = 0
        for i, data in enumerate(data_loader):
            if i != candidate_shape_indices[counter]:
                continue
            feats, label = data 
            feats = feats.to(device)
            label = label.to(device)
            feats = torch.squeeze(feats, dim=1) # feats: (B, 256, H, 1)
            
            with torch.no_grad():
                batch_ssa_feats, _ = self.get_ssa_feats(feats, 'test') # batch_ssa_feats: (B, 256, H, 1)
        
            if isinstance(ssa_feats, type(None)):
                ssa_feats = batch_ssa_feats.detach()
            else:
                ssa_feats = torch.concat((ssa_feats, batch_ssa_feats.detach()), dim=0) # ssa_feats: (N, 256, H, 1)

            counter += 1
            if counter == len(candidate_shape_indices):
                break
        ssa_feats = torch.permute(torch.squeeze(ssa_feats, dim=-1), (0,2,1)) # ssa_feats: (N, H, 256)
        return ssa_feats

    def get_retrieval_measure_big(self, query_loader, candidate_loader, candidate_shape_indices):
        candidate_shape_indices.sort()
        candidate_ssa_feats = self.get_candidate_ssa_feats(candidate_loader, candidate_shape_indices)

        ret_measure = None
        
        for qidx, qdata in enumerate(query_loader):
            feats, label = qdata 
            feats = feats.to(self.device)
            feats = torch.squeeze(feats, dim=1)

            with torch.no_grad():
                f1, _ = self.get_ssa_feats(feats, 'test')
            f1 = f1.squeeze(dim=-1).permute((0,2,1)).squeeze(dim=0)
            f1 = F.normalize(f1, dim=-1)

            f1_ret = None
            for f2 in candidate_ssa_feats:
                f2 = F.normalize(f2, dim=-1)
                cos_score = torch.matmul(f1.unsqueeze(0).unsqueeze(1), f2.unsqueeze(0).permute(0,2,1))
                r = cos_score.max(-1)[0].mean(-1)
                if isinstance(f1_ret, type(None)):
                    f1_ret = r.clone()
                else:
                    f1_ret = torch.cat((f1_ret, r), dim=1)
            if isinstance(ret_measure, type(None)):
                ret_measure = f1_ret.clone()
            else:
                ret_measure = torch.cat((ret_measure, f1_ret.clone()), dim=0)
        
        candidate_ssa_feats = None 

        return ret_measure

    def get_knn_graph_big(self, query_loader, candidate_loader, candidate_shape_indices, K):
        
        '''
            (ssa_feats_1, ssa_feats_2): ssa_feats of test shape, train shape (while testing)
                                        or both ssa_feats of train shape (while training)
        '''
        retrieval_measure = self.get_retrieval_measure_big(query_loader, candidate_loader, candidate_shape_indices)
        scores, knn_graph = retrieval_measure.topk(K+1, -1) # K+1: includes other shape and itself.
        retrieval_measure = None 
        scores = None
        return knn_graph

def backbone_ssa_fc_logit(num_classes, n_heads):
    d_model = 928
    model = CrossShapeAt(num_classes, d_model, n_heads, attention_type='ssa', after_fc=False)
    return model

def backbone_fc_ssa_logit(num_classes, n_heads):
    d_model = 256
    model = CrossShapeAt(num_classes, d_model, n_heads, attention_type='ssa', after_fc=True)
    return model

def backbone_csa_fc_logit(num_classes, n_heads, K):
    d_model = 928
    model = CrossShapeAt(num_classes, d_model, n_heads, K, attention_type='csa', after_fc=False)
    return model

def backbone_fc_csa_logit(num_classes, n_heads, K):
    d_model = 256
    model = CrossShapeAt(num_classes, d_model, n_heads, K, attention_type='csa', after_fc=True)
    return model

def get_model(attention_type, num_classes, n_heads, K=None):
    if attention_type == 'ssa':
        return backbone_fc_ssa_logit(num_classes, n_heads)
    elif attention_type == 'csa':
        return backbone_fc_csa_logit(num_classes, n_heads, K)
    else:
        raise AttributeError(f'{attention_type} not supported')
    