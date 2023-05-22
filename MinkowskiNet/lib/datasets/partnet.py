import logging
import os
import sys
import json
from pathlib import Path
import numpy as np
from lib.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.utils import read_txt


NUM_SEG = {'Bed': 15,
		   'Bottle': 9,
		   'Chair': 39,
		   'Clock': 11,
		   'Dishwasher': 7,
		   'Display': 4,
		   'Door': 5,
		   'Earphone': 10,
		   'Faucet': 12,
		   'Knife': 10,
		   'Lamp': 41,
		   'Microwave': 6,
		   'Refrigerator': 7,
		   'StorageFurniture': 24,
		   'Table': 51,
		   'TrashCan': 11,
		   'Vase': 6}


class PartnetVoxelizationDataset(VoxelizationDataset):

	# Voxelization arguments
	VOXEL_SIZE = 0.05

	# Augmentation arguments
	ROTATION_AUGMENTATION_BOUND = (-5 * np.pi / 180.0, 5 * np.pi / 180)
	JITTER_AUGMENTATION_BOUND = (0.25, 0.25, 0.25)
	SCALE_AUGMENTATION_BOUND = (0.75, 1.25)
	SHIFT_PARAMS = (0.01, 0.05) # ((sigma, clip)
	N_ROTATIONS = 1

	ROTATION_AXIS = 'y'

	DATA_PATH_FILE = {
		DatasetPhase.Train: 'train_files.txt',
		DatasetPhase.Val: 'val_files.txt',
		DatasetPhase.Test: 'test_files.txt'
	}

	def __init__(self,
				 config,
				 prevoxel_transform=None,
				 rot_aug=False,
				 phase=DatasetPhase.Train):

		# Init labels and color map
		self.NUM_LABELS = NUM_SEG[config.partnet_category.split('-')[0]]
		self.VALID_CLASS_IDS = tuple(range(self.NUM_LABELS))
		# in case we want to remove the zero label
		self.IGNORE_LABELS = tuple(set(range(self.NUM_LABELS)) - set(self.VALID_CLASS_IDS))
		self.partnet_category = config.partnet_category

		if isinstance(phase, str):
			phase = str2datasetphase_type(phase)
		self.phase = phase
		data_root = os.path.join(config.partnet_path, config.partnet_category)
		data_paths = read_txt(os.path.join(data_root, self.DATA_PATH_FILE[phase]))
		logging.info('Loading {} - {}: {}'.format(self.partnet_category, self.__class__.__name__,
												  self.DATA_PATH_FILE[phase]))
		self.input_feat = config.input_feat.lower()
		if self.input_feat == 'xyz':
			self.NUM_IN_CHANNEL = 3
		else:
			raise ValueError("Unknown input features {feat:s}" .format(feat=self.input_feat))
		if 'xyz' in self.input_feat:
			self.AUGMENT_COORDS_TO_FEATS = True

		super().__init__(
			data_paths,
			data_root=data_root,
			prevoxel_transform=prevoxel_transform,
			ignore_label=config.ignore_label,
			return_transformation=config.return_transformation,
			rot_aug=rot_aug,
			config=config)


class PartnetVoxelization0_05Dataset(PartnetVoxelizationDataset):
	VOXEL_SIZE = 0.05


class PartnetVoxelization0_04Dataset(PartnetVoxelizationDataset):
	VOXEL_SIZE = 0.04


class PartnetVoxelization0_03Dataset(PartnetVoxelizationDataset):
	VOXEL_SIZE = 0.03


class PartnetVoxelization0_02Dataset(PartnetVoxelizationDataset):
	VOXEL_SIZE = 0.02


class PartnetVoxelization0_01Dataset(PartnetVoxelizationDataset):
	VOXEL_SIZE = 0.01


class PartnetVoxelization0_005Dataset(PartnetVoxelizationDataset):
	VOXEL_SIZE = 0.005
