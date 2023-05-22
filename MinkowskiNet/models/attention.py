import numpy as np
import torch
from torch.nn.modules import Module
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me


from models.model import Model
from models.modules.common import NormType, get_norm
from models.modules.resnet_block import BasicBlock, Bottleneck
from lib.utils import features_at


class CSNet(Model):
  BLOCK = BasicBlock
  NUM_BLOCKS = 3
  PLANES = (64, 64, 128, 256)
  FC_PLANES = (1024, 512, 256, 128)
  NORM_TYPE = NormType.BATCH_NORM

  def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
    assert self.BLOCK is not None

    super(CSNet, self).__init__(in_channels, out_channels, config, D, **kwargs)

    self.network_initialization(in_channels, out_channels, config, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, out_channels, config, D):
    # Setup net_metadata
    bn_momentum = config.bn_momentum

    self.relu = ME.MinkowskiReLU(inplace=True)

    # MinkResNet backbone and CSA modules
    self.blocks = nn.ModuleList([])
    self.sim = nn.ModuleList([])
    self.linear_q = nn.ModuleList([])
    self.linear_k = nn.ModuleList([])
    self.linear_v = nn.ModuleList([])
    self.inplanes = in_channels
    for plane in self.PLANES:
      self.blocks.append(
                    self._make_layer(
                      block=self.BLOCK,
                      planes=plane,
                      blocks=self.NUM_BLOCKS,
                      stride=1,
                      dilation=1,
                      norm_type=self.NORM_TYPE,
                      bn_momentum=bn_momentum))
      self.inplanes = plane
      self.sim.append(ScaledDotProduct(plane**0.5))
      self.linear_q.append(ME.MinkowskiLinear(plane, plane, bias=False))
      self.linear_k.append(ME.MinkowskiLinear(plane, plane, bias=False))
      self.linear_v.append(ME.MinkowskiLinear(plane, plane, bias=False))

    # FC layer
    in_feat = np.sum(self.PLANES) * 2
    modules = []
    for out_feat in self.FC_PLANES:
      modules.append(ME.MinkowskiConvolution(
        in_feat,
        out_feat,
        kernel_size=1,
        bias=True,
        dimension=D))
      modules.append(get_norm(norm_type=self.NORM_TYPE, n_channels=out_feat, D=D, bn_momentum=bn_momentum))
      modules.append(self.relu)
      # modules.append(ME.MinkowskiDropout(0.5))
      in_feat = out_feat
    modules.append(ME.MinkowskiConvolution(
      self.FC_PLANES[-1],
      out_channels,
      kernel_size=1,
      bias=True,
      dimension=D))
    self.final = nn.Sequential(*modules)
    self.softmax = nn.Softmax(dim=1)

  def forward_backbone(self, x):
    out = [self.blocks[0](x)]
    for idx in range(1, len(self.blocks)):
      out.append(self.blocks[idx](out[idx-1]))

    return out

  def forward(self, queries, keys, comp, K):
    # Get features from MinkResNet backbone
    queries_out = self.forward_backbone(queries)
    keys_out = [queries_out]
    if K > 0:
      for idx in range(K):
        keys_out.append(self.forward_backbone(keys[idx]))

    # Calculate query, key and value transformations
    linear_queries_out, linear_keys_out, linear_values_out = [], [], []
    for i in range(len(self.linear_q)):
      linear_queries_out.append(self.linear_q[i](queries_out[i]))
      linear_keys_out.append([])
      linear_values_out.append([])
      for j in range(len(keys_out)):
        linear_keys_out[i].append(self.linear_k[i](keys_out[j][i]))
        linear_values_out[i].append(self.linear_v[i](keys_out[j][i]))

    # Get CSA features
    csa_queries_out = []
    batch_size = queries.C[-1, 0] + 1
    for b_idx in range(batch_size):
      csa_query = []
      for i in range(len(linear_queries_out)):
        # Get linear query features for query shape b_idx
        linear_query_feat = features_at(linear_queries_out[i], b_idx)
        csa = 0
        # linear_query_feat = linear_queries_out[i].features_at(b_idx) -> changes order of coordinates (fixed in newer versions)
        for j in range(len(linear_keys_out[i])):
          # Get linear key and value features for j-th key shape for current query shape
          linear_key_feat = features_at(linear_keys_out[i][j], b_idx)
          linear_value_feat = features_at(linear_values_out[i][j], b_idx)
          # Calculate attention
          attn = self.sim[i](linear_query_feat, linear_key_feat)
          attn = self.softmax(attn.squeeze(0))
          # Multiply attention matrix with values
          attn_values = torch.matmul(attn, linear_value_feat)
          # Multiply by shape compatibility
          csa += comp.F[b_idx, j] * attn_values
        csa_query.append(csa)
      csa_queries_out.append(torch.cat(csa_query, 1))

    csa_queries_out = torch.cat(csa_queries_out, 0)
    csa_queries_out = ME.SparseTensor(
      csa_queries_out,
      coordinate_map_key=queries.coordinate_map_key,
      coordinate_manager=queries.coordinate_manager)
    queries_out = me.cat(*queries_out)
    out = me.cat(queries_out, csa_queries_out)

    return self.final(out)

  def weight_initialization(self):
    for m in self.modules():
      if isinstance(m, ME.MinkowskiBatchNorm):
        nn.init.constant_(m.bn.weight, 1)
        nn.init.constant_(m.bn.bias, 0)

  def _make_layer(self,
                  block,
                  planes,
                  blocks,
                  stride=1,
                  dilation=1,
                  norm_type=NormType.BATCH_NORM,
                  bn_momentum=0.1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        ME.MinkowskiConvolution(
          self.inplanes,
          planes * block.expansion,
          stride=stride,
          dimension=self.D),
        get_norm(norm_type, planes * block.expansion, D=self.D, bn_momentum=bn_momentum)
      )
    layers = []
    layers.append(
      block(
        self.inplanes,
        planes,
        stride=stride,
        dilation=dilation,
        downsample=downsample,
        D=self.D,
        bn_momentum=bn_momentum))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(
        block(
          self.inplanes,
          planes,
          stride=1,
          dilation=dilation,
          D=self.D,
          bn_momentum=bn_momentum))

    return nn.Sequential(*layers)


class MultiHeadAttention(Module):
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

  def forward(self, q, k, v):
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

    return q, attn


class ScaledDotProductAttention(Module):
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


class ScaledDotProduct(Module):
  """
    Scaled dot-product
  """

  def __init__(self, temperature):
    super(ScaledDotProduct, self).__init__()

    self.temperature = temperature

  def forward(self, q, k):
    sparse = False
    if isinstance(q, ME.SparseTensor):
      assert(isinstance(k, ME.SparseTensor))
      output = torch.matmul(q.F, k.F.T)
      sparse = True
    else:
      if q.ndim == 2:
        q = q.unsqueeze(0)
      if k.ndim == 2:
        k = k.unsqueeze(0)
      output = torch.bmm(q, k.permute(0, 2, 1))

    output /= self.temperature

    if sparse:
      return ME.SparseTensor(
        output,
        coordinate_map_key=q.coordinate_map_key,
        coordinate_manager=q.coordinate_manager)
    return output

  def __repr__(self):
    s = '(temperature={})'.format(self.temperature)

    return self.__class__.__name__ + s
