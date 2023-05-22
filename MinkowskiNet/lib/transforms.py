import logging
import numpy as np
import torch

import MinkowskiEngine as ME

np.seterr(divide='raise')
_THRESHOLD_TOL_32 = 2.0 * np.finfo(np.float32).eps
_THRESHOLD_TOL_64 = 2.0 * np.finfo(np.float64).eps


class RandomShift:
	def __init__(self, sigma=0.01, clip=0.05):
		"""
		  Randomly shift the shape
		"""
		assert(clip > 0)
		self.sigma = sigma
		self.clip = clip

	def __call__(self, coords, feats, labels):
		bb_diagonal = np.array([np.amax(coords[:, 0]), np.amax(coords[:, 1]), np.amax(coords[:, 2])]) - \
					  np.array([np.amin(coords[:, 0]), np.amin(coords[:, 1]), np.amin(coords[:, 2])])
		bb_diagonal_length = np.sqrt(np.sum(bb_diagonal ** 2))
		std = self.sigma * bb_diagonal_length
		shift = np.clip(std * np.random.randn(1, 3), -1 * self.clip, self.clip)
		shifted_coords = coords + shift

		return shifted_coords, feats, labels


class RandomJittering:
	def __init__(self, x_jitter=0.01, y_jitter=0.01, z_jitter=0.01):
		"""
		  Randomly jitter points. Jittering is per axis.
		"""
		self.jitter = (x_jitter, y_jitter, z_jitter)

	def __call__(self, coords, feats, labels):
		jittered_coords = np.zeros((1,3))
		for i in range(jittered_coords.shape[1]):
			jittered_coords[0, i] = np.random.uniform(-self.jitter[i], self.jitter[i])
		jittered_coords = jittered_coords + coords

		return jittered_coords, feats, labels


class RandomScaling:
	def __init__(self, scale_lo=0.9, scale_up=1.1):
		"""
		  Randomly scale points. Scaling is uniform.
		"""
		self.scale_lo = scale_lo
		self.scale_up = scale_up

	def __call__(self, coords, feats, labels):
		scale_factor = np.random.uniform(self.scale_lo, self.scale_up)
		scale = np.zeros((3,3))
		np.fill_diagonal(scale, scale_factor)
		scaled_coords = np.matmul(scale, coords.T).T

		return scaled_coords, feats, labels


class RotationAugmentation:
	ANGLE = 0

	def __init__(self, use_normals=False):
		"""
		  Rotate the point cloud along up direction
		"""
		self.use_normals = use_normals

	@classmethod
	def update_angle(cls, angle):
		cls.ANGLE = angle

	def __call__(self, coords, feats, labels):
		cosval = np.cos(RotationAugmentation.ANGLE)
		sinval = np.sin(RotationAugmentation.ANGLE)
		rotation_matrix = np.array([[cosval, 0, sinval],
									[0, 1, 0],
									[-sinval, 0, cosval]])
		rot_coords = np.matmul(rotation_matrix, coords.T).T
		rot_feats = np.copy(feats)
		if self.use_normals:
			rot_feats[:, 0:3] = np.matmul(rotation_matrix, rot_feats[:, 0:3].T).T

		return rot_coords, rot_feats, labels


class Compose(object):
	"""Composes several transforms together."""

	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, *args):
		for t in self.transforms:
			args = t(*args)
		return args


class cfl_collate_fn_factory:
	"""Generates collate function for coords, feats, labels.

	  Args:
		limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
						 size so that the number of input coordinates is below limit_numpoints.
	"""

	def __init__(self, limit_numpoints, return_neighbors=False):
		self.limit_numpoints = limit_numpoints
		self.return_neighbors = return_neighbors

	def __call__(self, list_data):
		if self.return_neighbors:
			coords, feats, labels, neighbors = list(zip(*list_data))
			neighbors_batch = []
		else:
			coords, feats, labels = list(zip(*list_data))
		coords_batch, feats_batch, labels_batch = [], [], []

		batch_id = 0
		batch_num_points = 0
		for batch_id, _ in enumerate(coords):
			num_points = coords[batch_id].shape[0]
			batch_num_points += num_points
			if self.limit_numpoints and batch_num_points > self.limit_numpoints:
				num_full_points = sum(len(c) for c in coords)
				num_full_batch_size = len(coords)
				logging.warning(
					f'\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points '
					f'limit. Truncating batch size at {batch_id} out of {num_full_batch_size} with {batch_num_points - num_points}.'
				)
				break
			# coords_batch.append(torch.from_numpy(coords[batch_id]).int()) -> change this if you are not using TensorField
			coords_batch.append(torch.from_numpy(coords[batch_id]).float())
			feats_batch.append(torch.from_numpy(feats[batch_id]).float())
			labels_batch.append(torch.from_numpy(labels[batch_id]).int())
			if self.return_neighbors:
				neighbors_batch.append(neighbors[batch_id])
			batch_id += 1

		# Concatenate all lists
		coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch,
																		  dtype=coords_batch[0].dtype)

		if self.return_neighbors:
			return coords_batch, feats_batch, labels_batch, neighbors_batch

		return coords_batch, feats_batch, labels_batch


class cflt_collate_fn_factory:
	"""Generates collate function for coords, feats, labels, point_clouds, transformations.

	  Args:
		limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
						 size so that the number of input coordinates is below limit_numpoints.
	"""

	def __init__(self, limit_numpoints, return_neighbors=False):
		self.limit_numpoints = limit_numpoints
		self.return_neighbors = return_neighbors

	def __call__(self, list_data):
		if self.return_neighbors:
			coords, feats, labels, transformations, neighbors = list(zip(*list_data))
		else:
			coords, feats, labels, transformations = list(zip(*list_data))
		cfl_collate_fn = cfl_collate_fn_factory(limit_numpoints=self.limit_numpoints,
												return_neighbors=self.return_neighbors)
		if self.return_neighbors:
			coords_batch, feats_batch, labels_batch, neighbors_batch = cfl_collate_fn(list(zip(coords, feats, labels,
																							   neighbors)))
		else:
			coords_batch, feats_batch, labels_batch = cfl_collate_fn(list(zip(coords, feats, labels)))
		num_truncated_batch = coords_batch[:, 0].max().item() + 1

		batch_id = 0
		transformations_batch = []
		for transformation in transformations:
			if batch_id >= num_truncated_batch:
				break
			transformations_batch.append(torch.from_numpy(transformation).float())
			batch_id += 1

		if self.return_neighbors:
			return coords_batch, feats_batch, labels_batch, transformations_batch, neighbors_batch

		return coords_batch, feats_batch, labels_batch, transformations_batch


def normalize_coords(coords, method="sphere"):
	""" Normalize coordinates """
	centroid = np.mean(coords, axis=0)
	centered_coords = coords - centroid

	if method.lower() == "sphere":
		radius = bounding_sphere_radius(coords=centered_coords)
	elif method.lower() == "box":
		radius = bounding_box_diagonal(coords=centered_coords)
	else:
		raise ValueError("Unknown normalization method {}" .format(method))

	radius = np.maximum(radius, _THRESHOLD_TOL_64 if radius.dtype==np.float64 else _THRESHOLD_TOL_32)

	return centered_coords / radius


def bounding_box_diagonal(coords):
	""" Return bounding box diagonal """
	bb_diagonal = np.array([np.amax(coords[:, 0]), np.amax(coords[:, 1]), np.amax(coords[:, 2])]) - \
				  np.array([np.amin(coords[:, 0]), np.amin(coords[:, 1]), np.amin(coords[:, 2])])
	bb_diagonal_length = np.sqrt(np.sum(bb_diagonal ** 2))

	return bb_diagonal_length


def bounding_sphere_radius(coords):
	""" Return bounding sphere radius """
	radius = np.max(np.sqrt(np.sum(coords ** 2, axis=1)))

	return radius
