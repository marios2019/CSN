from enum import Enum
import torch.nn as nn

import MinkowskiEngine as ME


class NormType(Enum):
	BATCH_NORM = 0
	INSTANCE_NORM = 1
	INSTANCE_BATCH_NORM = 2
	LAYER_NORM = 3


def get_norm(norm_type, n_channels, D, bn_momentum=0.1, eps=1e-5):
	if norm_type == NormType.BATCH_NORM:
		return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
	elif norm_type == NormType.INSTANCE_NORM:
		return ME.MinkowskiInstanceNorm(n_channels)
	elif norm_type == NormType.INSTANCE_BATCH_NORM:
		return nn.Sequential(
			ME.MinkowskiInstanceNorm(n_channels),
			ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum))
	elif norm_type == NormType.LAYER_NORM:
		return MinkowskiLayerNorm(n_channels, eps)
	else:
		raise ValueError(f'Norm type: {norm_type} not supported')


class MinkowskiLayerNorm(nn.Module):
	"""A Layer normalization layer for a sparse tensor.
	Only normalization per channel is currently supported

	See the pytorch :attr:`torch.nn.LayerNorm` for more details.
	"""

	def __init__(self, num_features, eps=1e-5, affine=True):
		super(MinkowskiLayerNorm, self).__init__()
		self.ln = nn.LayerNorm(num_features, eps=eps, elementwise_affine=affine)

	def forward(self, input):
		output = self.ln(input.F)
		if isinstance(input, ME.TensorField):
			return ME.TensorField(
				output,
				coordinate_field_map_key=input.coordinate_field_map_key,
				coordinate_manager=input.coordinate_manager,
				quantization_mode=input.quantization_mode,
			)
		else:
			return ME.SparseTensor(
				output,
				coordinate_map_key=input.coordinate_map_key,
				coordinate_manager=input.coordinate_manager,
			)

	def __repr__(self):
		s = "({}, eps={}, affine={})".format(self.ln.normalized_shape, self.ln.eps, self.ln.elementwise_affine)
		return self.__class__.__name__ + s
