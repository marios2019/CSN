from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import h5py
import logging

import random
import numpy as np
from enum import Enum

import torch
from torch.utils.data import Dataset, DataLoader

import MinkowskiEngine as ME

import lib.transforms as t
from lib.dataloader import InfSampler
from lib.voxelizer import Voxelizer


class DatasetPhase(Enum):
	Train = 0
	Val = 1
	Val2 = 2
	TrainVal = 3
	Test = 4


def datasetphase_2str(arg):
	if arg == DatasetPhase.Train:
		return 'train'
	elif arg == DatasetPhase.Val:
		return 'val'
	elif arg == DatasetPhase.Val2:
		return 'val2'
	elif arg == DatasetPhase.TrainVal:
		return 'trainval'
	elif arg == DatasetPhase.Test:
		return 'test'
	else:
		raise ValueError('phase must be one of dataset enum.')


def str2datasetphase_type(arg):
	if arg.upper() == 'TRAIN':
		return DatasetPhase.Train
	elif arg.upper() == 'VAL':
		return DatasetPhase.Val
	elif arg.upper() == 'VAL2':
		return DatasetPhase.Val2
	elif arg.upper() == 'TRAINVAL':
		return DatasetPhase.TrainVal
	elif arg.upper() == 'TEST':
		return DatasetPhase.Test
	else:
		raise ValueError('phase must be one of train/val/test')


class VoxelizationDatasetBase():
	ROTATION_AXIS = None
	NUM_IN_CHANNEL = None
	NUM_LABELS = -1  # Number of labels in the dataset, including all ignore classes
	IGNORE_LABELS = None  # labels that are not evaluated
	IS_ONLINE_VOXELIZATION = True
	N_ROTATIONS = 1

	def __init__(self,
				 data_paths,
				 prevoxel_transform=None,
				 prefetch_data=False,
				 data_root='/',
				 ignore_mask=255,
				 return_transformation=False,
				 rot_aug=False,
				 normalize=False,
				 normalize_method="sphere",
				 input_feat='xyz',
				 load_h5=False,
				 return_neighbors=False,
				 **kwargs):
		"""
		ignore_mask: label value for ignore class. It will not be used as a class in the loss or evaluation.
		"""

		# Allows easier path concatenation
		if not isinstance(data_root, Path):
			data_root = Path(data_root)

		self.data_root = data_root
		self.data_paths = data_paths

		self.prevoxel_transform = prevoxel_transform

		self.ignore_mask = ignore_mask
		self.return_transformation = return_transformation
		self.rot_aug = rot_aug
		self.prefetch_data = prefetch_data
		self.normalize = normalize
		self.normalize_method = normalize_method
		self.input_feat = input_feat
		self.load_h5 = load_h5
		self.return_neighbors = return_neighbors

		if self.prefetch_data:
			self.prefetched_coords, self.prefetched_feats, self.prefetched_labels = [], [], []
			if load_h5:
				for data_ind in tqdm(range(len(self.data_paths))):
					coords, feats, labels = self.load_h5_files(data_ind, self.input_feat)
					if self.normalize:
						for i in range(coords.shape[0]):
							coords[i] = t.normalize_coords(coords[i], self.normalize_method)
					self.prefetched_coords.append(coords)
					self.prefetched_feats.append(feats)
					self.prefetched_labels.append(labels)
				self.prefetched_coords = np.vstack(self.prefetched_coords)
				self.prefetched_feats = np.vstack(self.prefetched_feats)
				self.prefetched_labels = np.vstack(self.prefetched_labels)
		else:
			raise ValueError("Need to prefetch data (--prefetch_data True")

		if self.rot_aug:
			angle = 2 * np.pi / self.N_ROTATIONS
			self.rotation_map = [(data_ind, rot_ind * angle) for data_ind in range(len(self.prefetched_coords))
								 for rot_ind in range(self.N_ROTATIONS)]
		if self.return_neighbors:
			self.neighbors = [(data_ind, []) for data_ind in range(len(self.prefetched_coords))]
		logging.info("#models: {}".format(self.__len__()))

	def __getitem__(self, index):
		raise NotImplementedError

	def load_h5_files(self, index, input_feat='xyz'):
		filepath = self.data_root / self.data_paths[index]
		# PartNet h5 format is used
		with h5py.File(filepath, 'r') as f_h5:
			# Load points
			coords = f_h5['data'][:].astype(np.float32)
			# Load features
			feats = np.full((coords.shape[0], 1), -1)
			label_key = 'label_seg'
			labels = np.squeeze(f_h5[label_key][:].astype(np.int32))

		if labels.ndim == 1:
			labels = labels[:, np.newaxis]

		return coords, feats, labels

	def __len__(self):
		num_data = len(self.data_paths)
		if self.prefetch_data:
			num_data = len(self.prefetched_coords)
		if self.rot_aug:
			num_data *= self.N_ROTATIONS
		return num_data


class VoxelizationDataset(VoxelizationDatasetBase):
	"""This dataset loads RGB point clouds and their labels as a list of points
	and voxelizes the pointcloud with sufficient data augmentation.
	"""
	# Voxelization arguments
	VOXEL_SIZE = 0.05  # 5cm

	# Augment coords to feats
	AUGMENT_COORDS_TO_FEATS = False

	def __init__(self,
				 data_paths,
				 prevoxel_transform=None,
				 data_root='/',
				 ignore_label=255,
				 return_transformation=False,
				 rot_aug=False,
				 config=None,
				 **kwargs):

		self.config = config

		VoxelizationDatasetBase.__init__(
			self,
			data_paths,
			prevoxel_transform=prevoxel_transform,
			prefetch_data=self.config.prefetch_data,
			data_root=data_root,
			ignore_mask=ignore_label,
			return_transformation=return_transformation,
			rot_aug=rot_aug,
			normalize=config.normalize_coords,
			normalize_method=config.normalize_method,
			input_feat=self.input_feat,
			load_h5=config.load_h5,
			return_neighbors=config.return_neighbors)

		# Prevoxel transformations
		self.voxelizer = Voxelizer(
			voxel_size=self.VOXEL_SIZE,
			ignore_label=ignore_label)

		# map labels not evaluated to ignore_label
		label_map = {}
		n_used = 0
		for l in range(self.NUM_LABELS):
			if l in self.IGNORE_LABELS:
				label_map[l] = self.ignore_mask
			else:
				label_map[l] = n_used
				n_used += 1
		label_map[self.ignore_mask] = self.ignore_mask
		self.label_map = label_map
		self.NUM_LABELS -= len(self.IGNORE_LABELS)

	def _augment_coords_to_feats(self, coords, feats, labels=None):
		if (feats == -1).all():
			feats = coords.copy()
		elif isinstance(coords, np.ndarray):
			feats = np.concatenate((coords, feats), 1)
		else:
			feats = torch.cat((coords, feats), 1)
		return coords, feats, labels

	def __getitem__(self, index):
		if self.rot_aug:
			if self.config.random_rotation:
				angle = np.random.uniform(self.ROTATION_AUGMENTATION_BOUND[0], self.ROTATION_AUGMENTATION_BOUND[1])
			else:
				index, angle = self.rotation_map[index]
			t.RotationAugmentation.update_angle(angle)
		coords = np.copy(self.prefetched_coords[index])
		feats = np.copy(self.prefetched_feats[index])
		labels = np.copy(self.prefetched_labels[index])

		# Prevoxel transformations
		if self.prevoxel_transform is not None:
			coords, feats, labels = self.prevoxel_transform(coords, feats, labels)

		# Use coordinate features
		if self.AUGMENT_COORDS_TO_FEATS:
			coords, feats, labels = self._augment_coords_to_feats(coords, feats, labels)

		coords, feats, labels, transformation = self.voxelizer.voxelize(coords, feats, labels)

		# map labels not used for evaluation to ignore_label
		if self.IGNORE_LABELS is not None:
			labels = np.array([self.label_map[x] for x in labels], dtype=np.int)

		return_args = [coords, feats, labels]
		if self.return_transformation:
			return_args.append(transformation.astype(np.float32))
		if self.return_neighbors:
			return_args.append(self.neighbors[index])

		return tuple(return_args)


def initialize_data_loader(DatasetClass,
						   config,
						   phase,
						   num_workers,
						   shuffle,
						   repeat,
						   shift,
						   jitter,
						   scale,
						   rot_aug,
						   batch_size,
						   limit_numpoints):
	if isinstance(phase, str):
		phase = str2datasetphase_type(phase)

	if config.return_transformation:
		collate_fn = t.cflt_collate_fn_factory(limit_numpoints, config.return_neighbors)
	else:
		collate_fn = t.cfl_collate_fn_factory(limit_numpoints, config.return_neighbors)

	prevoxel_transform_train = []
	if rot_aug:
		prevoxel_transform_train.append(t.RotationAugmentation(True if 'normals' in config.input_feat else False))
	if shift:
		prevoxel_transform_train.append(t.RandomShift(*DatasetClass.SHIFT_PARAMS))
	elif jitter:
		prevoxel_transform_train.append(t.RandomJittering(*DatasetClass.JITTER_AUGMENTATION_BOUND))
	if scale:
		prevoxel_transform_train.append(t.RandomScaling(*DatasetClass.SCALE_AUGMENTATION_BOUND))

	if len(prevoxel_transform_train) > 0:
		prevoxel_transforms = t.Compose(prevoxel_transform_train)
	else:
		prevoxel_transforms = None

	dataset = DatasetClass(
		config,
		prevoxel_transform=prevoxel_transforms,
		rot_aug=rot_aug,
		phase=phase)

	data_args = {
		'dataset': dataset,
		'num_workers': num_workers,
		'batch_size': batch_size,
		'collate_fn': collate_fn,
	}

	if repeat:
		data_args['sampler'] = InfSampler(dataset, shuffle)
	else:
		data_args['shuffle'] = shuffle

	data_loader = DataLoader(**data_args)

	return data_loader
