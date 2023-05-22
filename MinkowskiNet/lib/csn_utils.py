import numpy as np
import torch
import logging
from copy import deepcopy
from tqdm import tqdm

from MinkowskiEngine import TensorField
from MinkowskiEngine.utils import sparse_collate


def construct_shape_graph(model, device, query_dataset, mink_settings, key_dataset=None, K=1, writer=None,
						  tag='', step=0, random_pairs=False):
	"""
		Construct K-nn shape graph for each shape in the query dataset
	"""
	assert(K != 0)

	query_dataset = deepcopy(query_dataset)
	if key_dataset is not None:
		key_dataset = deepcopy(key_dataset)
		is_same = False
	else:
		key_dataset = deepcopy(query_dataset)
		is_same = True
	# Disable data augmentations
	query_dataset.prevoxel_transform = None
	key_dataset.prevoxel_transform = None

	query_neighbors = []

	if random_pairs:
		# Get k-nn randomly
		logging.info("===> Get random pairs")
		for idx in range(len(query_dataset)):
			indices = np.random.choice(len(key_dataset), K, replace=False)
			if is_same:
				# In case the query shape is randomly selected as one of its neighbors
				while(1):
					if idx not in indices:
						break
					indices = np.random.choice(len(key_dataset), K, replace=False)
			neighbors = indices.tolist()
			query_neighbors.append((idx, neighbors))
	else:
		logging.info("===> Get pairs based on cosine similarity (SSA)")
		keys_ssa = {}
		with torch.no_grad():
			# Get similarity
			for q_idx in tqdm(range(len(query_dataset)), desc="Find neighbors (K={})".format(K)):
				# Query shape
				query_coords, query_input, _, _ = query_dataset[q_idx]
				query_coords = torch.from_numpy(query_coords).float()
				query_input = torch.from_numpy(query_input).float()
				query_coords_batch, query_input = sparse_collate([query_coords], [query_input], dtype=query_coords.dtype)
				query_field = TensorField(
					features=query_input,
					coordinates=query_coords_batch,
					quantization_mode=mink_settings["q_mode"],
					minkowski_algorithm=mink_settings["mink_algo"],
					device=device)
				query_sparse = query_field.sparse()
				# Feed forward
				query_ssa_feat = model(query_sparse, return_ssa=True).F
				similarity = []
				for k_idx in range(len(key_dataset)):
					if k_idx in keys_ssa.keys():
						key_ssa_feat = keys_ssa[k_idx].cuda()
					else:
						# Key shape
						key_coords, key_input, _, _ = key_dataset[k_idx]
						key_coords = torch.from_numpy(key_coords).float()
						key_input = torch.from_numpy(key_input).float()
						key_coords_batch, key_input = sparse_collate([key_coords], [key_input], dtype=key_coords.dtype)
						key_field = TensorField(
							features=key_input,
							coordinates=key_coords_batch,
							quantization_mode=mink_settings["q_mode"],
							minkowski_algorithm=mink_settings["mink_algo"],
							device=device)
						key_sparse = key_field.sparse()
						# Feed forward
						key_ssa_feat = model(key_sparse, return_ssa=True).F
						keys_ssa[k_idx] = key_ssa_feat.cpu()
					# Calculate similarity
					sim = model.cosine_similarity(query_ssa_feat, key_ssa_feat)
					similarity.append(sim)
					# Clear cache
					torch.cuda.empty_cache()

				# Get k-nn shapes based on similarity
				similarity = torch.stack(similarity)
				vals, indices = torch.topk(similarity, K)
				if is_same and q_idx in indices:
					vals, indices = torch.topk(similarity, K + 1)
					indices = indices[q_idx != indices]
				neighbors = indices.tolist()
				query_neighbors.append((q_idx, neighbors))

	if writer is not None:
		# Log first 2 point clouds along with their neighbors
		log_pc = 2 if len(query_dataset) >= 2 else len(query_dataset)
		for idx in range(log_pc):
			query_pc = query_dataset.prefetched_coords[idx]
			neighbors = query_neighbors[idx][1]
			writer.add_mesh(tag+'/query_pc_'+str(idx), vertices=query_pc[np.newaxis, ...], global_step=step)
			for nn_idx in neighbors:
				neighbor_pc = key_dataset.prefetched_coords[nn_idx]
				writer.add_mesh(tag + '/query_pc_' + str(idx) + '/neighbor_pc_' + str(nn_idx),
				                vertices=neighbor_pc[np.newaxis, ...], global_step=step)

	return query_neighbors


def get_neighbors(key_dataset, neighbors, K):
	"""
		Get K neighbors for query shapes from key_dataset
	"""

	query_neighbors = []

	for i in range(K):
		coords_batch, input_batch = [], []
		for neighborhood in neighbors:
			coords, input, _, _ = key_dataset[neighborhood[1][i]]
			coords_batch.append(torch.from_numpy(coords).float())
			input_batch.append(torch.from_numpy(input).float())
		coords_batch, input_batch = sparse_collate(coords_batch, input_batch, dtype=coords_batch[0].dtype)
		query_neighbors.append([input_batch, coords_batch])

	return query_neighbors
