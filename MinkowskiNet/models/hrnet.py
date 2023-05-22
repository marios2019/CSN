import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me

from models.model import Model
from models.modules.common import NormType, get_norm
from models.modules.resnet_block import BasicBlock
from models.attention import ScaledDotProduct, MultiHeadAttention
from lib.utils import features_at


class HRNetBase(Model):
	BLOCK = None
	NUM_STAGES = 1
	NUM_BLOCKS = 3
	INIT_DIM = 32
	FEAT_FACTOR = 1
	NORM_TYPE = NormType.BATCH_NORM

	def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
		assert self.BLOCK is not None

		super(HRNetBase, self).__init__(in_channels, out_channels, config, D, **kwargs)

		self.backbone_initialization(in_channels, config, D)

	def backbone_initialization(self, in_channels, config, D):
		# Setup net_metadata
		bn_momentum = config.bn_momentum

		self.relu = ME.MinkowskiReLU(inplace=True)

		# Initial input features transformation -> concat with hrnet output
		self.inplanes = self.INIT_DIM
		self.conv0s1 = ME.MinkowskiConvolution(
			in_channels,
			self.inplanes,
			kernel_size=config.conv1_kernel_size,
			dimension=D)
		self.bn0s1 = get_norm(norm_type=self.NORM_TYPE, n_channels=self.inplanes, D=D, bn_momentum=bn_momentum)

		# Create high-to-low resolution branches
		self.init_stage_dims = self.INIT_DIM * self.FEAT_FACTOR
		self.conv1s1 = ME.MinkowskiConvolution(
			self.inplanes,
			self.init_stage_dims,
			kernel_size=3,
			dimension=D)
		self.bn1s1 = get_norm(norm_type=self.NORM_TYPE, n_channels=self.init_stage_dims, D=D, bn_momentum=bn_momentum)
		self.stages, self.exchange_blocks = nn.ModuleList([]), nn.ModuleList([])
		for i in range(self.NUM_STAGES):
			# Each stage includes #stage branches
			stage = nn.ModuleList([])
			for j in range(i + 1):
				self.inplanes = self.init_stage_dims * 2 ** j
				stage.append(
					self._make_layer(
						block=self.BLOCK,
						planes=self.init_stage_dims * 2 ** j,
						blocks=self.NUM_BLOCKS,
						stride=1,
						dilation=1,
						norm_type=self.NORM_TYPE,
						bn_momentum=bn_momentum))
			self.stages.append(stage)

			# Create exchange blocks
			if i == (self.NUM_STAGES - 1):
				# No exchange blocks for the last stage
				break
			exchange_blocks = nn.ModuleList([])
			depth = len(stage)
			for j in range(depth):
				exchange_block = nn.ModuleList([])
				init_channels = self.init_stage_dims * 2 ** j
				for k in range(depth + 1):
					d0, d1 = depth - j, depth - k
					block = nn.ModuleList([])
					add_relu = False
					if d0 > d1:
						# Downsampling
						for s in range(d0 - d1):
							if add_relu:
								block.append(ME.MinkowskiReLU())
							block.append(
								ME.MinkowskiConvolution(
									int(init_channels * 2 ** s),
									int(init_channels * 2 ** (s + 1)),
									kernel_size=3,
									stride=2,
									dimension=D))
							block.append(
								get_norm(norm_type=self.NORM_TYPE, n_channels=init_channels * 2 ** (s + 1),
										 D=D, bn_momentum=bn_momentum))
							add_relu = True
					elif d0 < d1:
						# Upsampling
						for s in range(0, -(d1 - d0), -1):
							if add_relu:
								block.append(ME.MinkowskiReLU())
							block.append(
								ME.MinkowskiConvolutionTranspose(
									int(init_channels * 2 ** s),
									int(init_channels * 2 ** (s - 1)),
									kernel_size=3,
									stride=2,
									dimension=D))
							block.append(
								get_norm(norm_type=self.NORM_TYPE, n_channels=int(init_channels * 2 ** (s - 1)),
										 D=D, bn_momentum=bn_momentum))
							add_relu = True
					else:
						block.append(nn.ModuleList([]))
					exchange_block.append(nn.Sequential(*block))
				exchange_blocks.append(exchange_block)
			self.exchange_blocks.append(exchange_blocks)

	def forward_backbone(self, x):
		# Initial input features transformation
		out = self.conv0s1(x)
		out = self.bn0s1(out)
		out_init = self.relu(out)

		# Feature transform to high-resolution branch
		out = self.conv1s1(out_init)
		out = self.bn1s1(out)
		out = self.relu(out)

		# Transform features through HRNet multi-resolution branches
		for i in range(self.NUM_STAGES):
			if i == 0:
				# Only for 1st stage
				stage_input = [out]
			stage_output = []
			for j in range(i + 1):
				stage_output.append(self.stages[i][j](stage_input[j]))
			if i == (self.NUM_STAGES - 1):
				# No exchange blocks for the last stage
				break
			stage_input = [[] for _ in range(len(self.stages[i + 1]))]
			m = len(stage_input)
			depth = len(stage_output)
			for j in range(depth):
				for k in range(depth + 1):
					if j < k:
						# Downsampling
						stage_input[k].append(self.exchange_blocks[i][j][k % m](stage_output[j]))
					elif j > k:
						# Upsampling
						stage_input[k].append(self.exchange_blocks[i][j][k % m](stage_output[j]))
					else:
						stage_input[k].append(stage_output[j])
			for j in range(len(stage_input)):
				buf = stage_input[j][0]
				for k in range(1, len(stage_input[j])):
					buf = buf + stage_input[j][k]
				stage_input[j] = self.relu(buf)

		return out_init, stage_output

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
					kernel_size=1,
					stride=stride,
					dimension=D),
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


class HRNetSeg(HRNetBase):

	def __init__(self, in_channels, out_channels, config, D=3, **kwargs):

		super(HRNetSeg, self).__init__(in_channels, out_channels, config, D, **kwargs)

		self.sem_seg_head_initialization(out_channels, D, config)
		self.weight_initialization()

	def sem_seg_head_initialization(self, out_channels, D, config):
		bn_momentum = config.bn_momentum

		# Final transitions
		self.final_transitions = nn.ModuleList([])
		for i in range(1, self.NUM_STAGES):
			# Upsample all lower resolution branches to the highest resolution branch
			init_channels = self.init_stage_dims * 2 ** i
			block = nn.ModuleList([])
			for j in range(i):
				block.append(
					ME.MinkowskiConvolutionTranspose(
						init_channels,
						init_channels,
						kernel_size=3,
						stride=2,
						dimension=D))
				block.append(
					get_norm(norm_type=self.NORM_TYPE, n_channels=init_channels,
							 D=D, bn_momentum=bn_momentum))
				block.append(ME.MinkowskiReLU())
			self.final_transitions.append(nn.Sequential(*block))

		# FC layer
		backbone_out_feat = np.sum([self.init_stage_dims * 2 ** s for s in range(self.NUM_STAGES)]) + self.INIT_DIM
		fc1 = ME.MinkowskiConvolution(
			backbone_out_feat,
			256,
			kernel_size=1,
			bias=True,
			dimension=D)
		bnfc1 = get_norm(norm_type=self.NORM_TYPE, n_channels=256, D=D, bn_momentum=bn_momentum)
		fc2 = ME.MinkowskiConvolution(
			256,
			out_channels,
			kernel_size=1,
			bias=True,
			dimension=D)

		self.final = nn.Sequential(fc1, bnfc1, self.relu, fc2)

	def forward(self, x):

		# Get features from HRNet backbone
		out_init, stage_output = self.forward_backbone(x)

		# Final transitions
		out = [out_init, stage_output[0]]
		for i in range(1, self.NUM_STAGES):
			out.append(self.final_transitions[i - 1](stage_output[i]))
		out = me.cat(*out)

		return self.final(out)


class HRNetSeg2S(HRNetSeg):
	BLOCK = BasicBlock
	FEAT_FACTOR = 2
	NUM_STAGES = 2


class HRNetSeg3S(HRNetSeg):
	BLOCK = BasicBlock
	FEAT_FACTOR = 2
	NUM_STAGES = 3


class HRNetSeg4S(HRNetSeg):
	BLOCK = BasicBlock
	FEAT_FACTOR = 2
	NUM_STAGES = 4


class HRNetSimCSN(HRNetBase):

	def __init__(self, in_channels, out_channels, config, D=3, **kwargs):

		super(HRNetSimCSN, self).__init__(in_channels, out_channels, config, D, **kwargs)

		self.csn_head_initialization(out_channels, D, config)
		self.weight_initialization()

	def csn_head_initialization(self, out_channels, D, config):
		bn_momentum = config.bn_momentum

		# Final transitions
		self.final_transitions = nn.ModuleList([])
		for i in range(1, self.NUM_STAGES):
			# Upsample all lower resolution branches to the highest resolution branch
			init_channels = self.init_stage_dims * 2 ** i
			block = nn.ModuleList([])
			for j in range(i):
				block.append(
					ME.MinkowskiConvolutionTranspose(
						init_channels,
						init_channels,
						kernel_size=3,
						stride=2,
						dimension=D))
				block.append(
					get_norm(norm_type=self.NORM_TYPE, n_channels=init_channels,
							 D=D, bn_momentum=bn_momentum))
				block.append(ME.MinkowskiReLU())
			self.final_transitions.append(nn.Sequential(*block))

		# FC layer
		backbone_out_feat = np.sum([self.init_stage_dims * 2 ** s for s in range(self.NUM_STAGES)]) + self.INIT_DIM

		self.d_model = config.d_model
		fc1 = ME.MinkowskiConvolution(
			backbone_out_feat,
			self.d_model,
			kernel_size=1,
			bias=True,
			dimension=D)
		bnfc1 = get_norm(norm_type=self.NORM_TYPE, n_channels=self.d_model, D=D, bn_momentum=bn_momentum)
		self.fc_layer = nn.Sequential(fc1, bnfc1, self.relu)

		# CSA layer
		self.n_head = config.n_head
		self.MHA = MultiHeadAttention(self.n_head, self.d_model, self.d_model // self.n_head, self.d_model // self.n_head)

		# Output layer
		self.output = ME.MinkowskiConvolution(
			self.d_model * 2,
			out_channels,
			kernel_size=1,
			bias=True,
			dimension=D)

		if self.config.k_neighbors > 0:
			# For similarity - compatibility
			self.linear_q = nn.Linear(self.d_model, self.d_model, bias=False)
			self.linear_k = nn.Linear(self.d_model, self.d_model, bias=False)
			self.sim = ScaledDotProduct(self.d_model ** 0.5)

	def forward(self, queries, keys=None, return_ssa=False):
		K = len(keys) if keys is not None else 0

		# Get queries and keys features from backbone
		queries_out, keys_out = self.backbone(queries, keys, K)

		# Get query SSA features
		queries_SSA = self.get_SSA(queries_out)
		if return_ssa:
			return queries_SSA

		if K > 0:
			csa_queries_out = []
			# Get key SSA features
			keys_SSA = [queries_SSA]
			for idx in range(K):
				keys_SSA.append(self.get_SSA(keys_out[idx]))
			# CSA layer
			batch_size = queries_out.C[-1, 0] + 1
			for b_idx in range(batch_size):
				query_ssa_feat = features_at(queries_SSA, b_idx)
				# Query SSA global rep (query linear transformation)
				query_ssa_avg_pool = torch.mean(query_ssa_feat, dim=0)
				query_ssa_glob = self.linear_q(query_ssa_avg_pool)
				query_ssa_glob = F.normalize(query_ssa_glob, dim=-1)
				similarity = []
				for i in range(len(keys_SSA)):
					key_ssa_feat = features_at(keys_SSA[i], b_idx)
					# Key SSA global rep (key linear transformation)
					key_ssa_avg_pool = torch.mean(key_ssa_feat, dim=0)
					key_ssa_glob = self.linear_k(key_ssa_avg_pool)
					key_ssa_glob = F.normalize(key_ssa_glob, dim=-1)
					# Calculate similarity between query-key
					sim = self.sim(query_ssa_glob.unsqueeze(0), key_ssa_glob.unsqueeze(0)).squeeze()
					similarity.append(sim)
				similarity = torch.stack(similarity)
				# Calculate compatibility between query-key
				comp = F.softmax(similarity, dim=0)
				# Multiply by self-shape compatibility
				csa = comp[0] * query_ssa_feat
				# Query backbone features
				query_feat = features_at(queries_out, b_idx)
				query_feat = query_feat.unsqueeze(0)
				for i in range(len(keys_out)):
					# Key backbone features
					key_feat = features_at(keys_out[i], b_idx)
					key_feat = key_feat.unsqueeze(0)
					# Get query-key CSA features
					csa_feat, _ = self.MHA(query_feat, key_feat, key_feat)
					# Multiply by cross-shape compatibility
					csa += comp[i + 1] * csa_feat.squeeze(0)
				csa_queries_out.append(csa)

			# Stack CSA for all shapes
			csa_queries_out = torch.cat(csa_queries_out, 0)
			csa_queries_out = ME.SparseTensor(
				csa_queries_out,
				coordinate_map_key=queries.coordinate_map_key,
				coordinate_manager=queries.coordinate_manager)
		else:
			csa_queries_out = queries_SSA

		out = me.cat(queries_out, csa_queries_out)

		return self.output(out)

	def backbone(self, queries, keys, K):
		# Get features from HRNet backbone
		queries_out_init, queries_stage_output = self.forward_backbone(queries)
		keys_out_backbone = []
		if K > 0:
			for idx in range(K):
				keys_out_backbone.append(self.forward_backbone(keys[idx]))

		# Final transitions for queries
		queries_out = [queries_out_init, queries_stage_output[0]]
		for i in range(1, self.NUM_STAGES):
			queries_out.append(self.final_transitions[i - 1](queries_stage_output[i]))
		queries_out = me.cat(*queries_out)
		# FC layer
		queries_out = self.fc_layer(queries_out)

		# Final transitions for keys
		keys_out = None
		if K > 0:
			keys_out = []
			for idx in range(K):
				key_out = [keys_out_backbone[idx][0], keys_out_backbone[idx][1][0]]
				for i in range(1, self.NUM_STAGES):
					key_out.append(self.final_transitions[i - 1](keys_out_backbone[idx][1][i]))
				key_out = me.cat(*key_out)
				# FC layer
				key_out = self.fc_layer(key_out)
				keys_out.append(key_out)

		return queries_out, keys_out

	def get_SSA(self, X):
		SSA = []
		batch_size = X.C[-1, 0] + 1
		for b_idx in range(batch_size):
			feat = features_at(X, b_idx)
			feat = feat.unsqueeze(0)
			# Calculate SSA
			ssa_feat, _ = self.MHA(feat, feat, feat)
			SSA.append(ssa_feat.squeeze(0))
		SSA = torch.cat(SSA, 0)

		return ME.SparseTensor(
			SSA,
			coordinate_map_key=X.coordinate_map_key,
			coordinate_manager=X.coordinate_manager)

	@staticmethod
	def cosine_similarity(q, k):
		# Normalize feature vectors per row
		q_length = torch.sum(q ** 2, dim=1) ** 0.5
		q_norm = q / q_length.unsqueeze(1)
		# Normalize feature vectors per row
		k_length = torch.sum(k ** 2, dim=1) ** 0.5
		k_norm = k / k_length.unsqueeze(1)
		# Calculate cosine similarity
		q_norm = q_norm.unsqueeze(0)
		k_norm = k_norm.unsqueeze(0)
		sim = torch.bmm(q_norm, k_norm.permute(0, 2, 1)).squeeze(0)

		# Max pooling per channel
		max_row, _ = torch.max(sim, 1)
		# Global avg pooling
		mean_val = torch.mean(max_row)

		return mean_val


class HRNetSimCSN2S(HRNetSimCSN):
	BLOCK = BasicBlock
	FEAT_FACTOR = 4
	NUM_STAGES = 2


class HRNetSimCSN3S(HRNetSimCSN):
	BLOCK = BasicBlock
	FEAT_FACTOR = 2
	NUM_STAGES = 3


class HRNetSimCSN4S(HRNetSimCSN):
	BLOCK = BasicBlock
	FEAT_FACTOR = 2
	NUM_STAGES = 4
