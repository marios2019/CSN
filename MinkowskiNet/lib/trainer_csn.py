import logging
import os.path as osp
import os
import numpy as np

import torch
from torch import nn

from tensorboardX import SummaryWriter

from lib.utils import checkpoint, precision_at_one_partnet, calculate_iou, calculate_shape_iou, calculate_part_iou, \
	set_grad, Timer, AverageMeter, get_torch_device
from lib.solvers import initialize_optimizer, initialize_scheduler
from lib.csn_utils import construct_shape_graph, get_neighbors

import MinkowskiEngine as ME


class Trainer():

	def __init__(self, model, data_loader, val_data_loader, config, mink_settings):

		# Configuration
		self.device = get_torch_device(config.is_cuda)
		self.model = model
		self.model_name = self.model.__class__.__name__
		self.data_loader = data_loader
		self.val_data_loader = val_data_loader
		self.config = config
		self.mink_settings = mink_settings

		self.writer = SummaryWriter(log_dir=self.config.log_dir)
		self.data_timer, self.iter_timer = Timer(), Timer()
		self.data_time_avg, self.iter_time_avg = AverageMeter(), AverageMeter()
		self.losses, self.scores = AverageMeter(), AverageMeter()
		self.MAX_PATIENCE, self.MAX_COOLDOWN, self.MAX_GRAPH_CONSTRUCTION = 10, 5, 3
		self.patience, self.cooldown = self.MAX_PATIENCE, self.MAX_COOLDOWN


		self.optimizer = initialize_optimizer(self.model.parameters(), self.config)
		self.lr_factor = 0.5
		self.scheduler = initialize_scheduler(self.optimizer, self.config, factor=self.lr_factor,
		                                      patience=self.MAX_PATIENCE, cooldown=self.MAX_COOLDOWN * 2,
		                                      verbose=True)
		self.criterion = nn.CrossEntropyLoss(ignore_index=self.config.ignore_label)

		self.best_val_part_iou, self.best_val_part_iou_iter = 0, 0
		self.best_val_shape_iou, self.best_val_shape_iou_iter = 0, 0
		self.best_val_loss, self.best_val_loss_iter = np.Inf, 0
		self.best_val_acc, self.best_val_acc_iter = 0, 0
		self.curr_iter, self.epoch, self.is_training = 1, 1, True
		self.n_graph_construction = 0

	def train(self):
		# Train the network
		self.model.train()
		logging.info('===> Start training')

		if self.config.resume:
			# Resume training
			self._resume()
			# Recalculate graph if needed, before resuming training
			if self.config.k_neighbors > 0 and self.patience <= 0:
				# Construct graph
				self.model.eval()
				self._construct_shape_graph(recalculate=True)
				self.model.train()
				# Reset patience
				self.n_graph_construction += 1
				self.patience = self.MAX_PATIENCE
				self.cooldown = self.MAX_COOLDOWN

		if self.config.save_param_histogram:
			# Log initial params histograms
			self._log_params()

		# Construct shape graph
		if self.config.k_neighbors > 0 and not self.config.resume:
			# Construct graph
			self.model.eval()
			self._construct_shape_graph(recalculate=False)
			self.n_graph_construction += 1
			self.model.train()

		self.data_iter = self.data_loader.__iter__()
		torch.autograd.set_detect_anomaly(True)

		# Number of batches
		self.data_len = (len(self.data_loader) + self.config.iter_size - 1) // self.config.iter_size

		while self.is_training:

			### START OF EPOCH
			for iteration in range(self.data_len):
				# Train for one iteration
				self._train_iter()

				if self.curr_iter % self.config.stat_freq == 0 or self.curr_iter == 1:
					# Log stats
					self._log_stats()

				# End of iteration
				self.curr_iter += 1
			### END OF EPOCH

			if self.epoch >= self.config.max_epoch:
				# Terminate training
				self.is_training = False
				break

			# Save current status, save before val to prevent occational mem overflow
			self._save_curr_checkpoint()

			# Validation
			self.cooldown -= 1
			val_loss, val_score, val_part_iou, val_shape_iou = self._validate()
			if val_part_iou > self.best_val_part_iou:
				# Reset patience
				self.patience = self.MAX_PATIENCE
			else:
				if self.config.k_neighbors > 0 and self.n_graph_construction < self.MAX_GRAPH_CONSTRUCTION:
					# Patience until change training mode
					if self.cooldown <= 0:
						self.cooldown = 0
						self.patience -= 1
						logging.info('=====> (Iteration:{}) Patience running out (patience:{})'
						             .format(self.curr_iter, self.patience))
					else:
						logging.info('=====> (Iteration:{}) Getting hotter (cooldown:{})'
						             .format(self.curr_iter, self.cooldown))
			self._save_best_checkpoints(val_loss, val_score, val_part_iou, val_shape_iou)
			self._log_val_stats()

			if self.config.k_neighbors > 0:
				# Recalculate graph
				if self.patience <= 0:
					# Load best_part_iou model
					checkpoint_fn = osp.join(self.config.log_dir, "checkpoint_{}best_part_iou.pth".format(self.config.model))
					logging.info("=====> Loading checkpoint '{}'".format(checkpoint_fn))
					state = torch.load(checkpoint_fn)
					self.model.load_state_dict(state['state_dict'])
					logging.info("=====> Checkpoint loaded from epoch {} (iter {})".format(state['epoch'], state['iteration']))
					if self.config.resume_optimizer:
						self.optimizer.load_state_dict(state['optimizer'])
						self.optimizer.param_groups[0]['lr'] = self.config.lr
						logging.info("===> Optimizer loaded")
						self.scheduler = \
							initialize_scheduler(self.optimizer, self.config, last_step=self.curr_iter, factor=self.lr_factor)
					# Construct graph
					self.model.eval()
					self._construct_shape_graph(recalculate=True)
					self.model.train()
					# Reset patience
					self.n_graph_construction += 1
					self.patience = self.MAX_PATIENCE
					self.cooldown = self.MAX_COOLDOWN
					# Save newly constructed graph
					self._save_curr_checkpoint()

			# Recover back
			self.model.train()

			if self.config.scheduler == "ReduceLROnPlateau":
				try:
					self.scheduler.step(val_loss)
				except UnboundLocalError:
					pass

			if self.config.save_param_histogram:
				# Log params histograms
				if self.epoch % self.config.param_histogram_freq == 0:
					self._log_params()
			self.losses.reset()
			self.scores.reset()
			self.epoch += 1

		# Explicit memory cleanup
		if hasattr(self.data_iter, 'cleanup'):
			self.data_iter.cleanup()

		# Save the final model
		val_loss, val_score, val_part_iou, val_shape_iou = self._validate()
		self._save_curr_checkpoint()
		self._save_best_checkpoints(val_loss, val_score, val_part_iou, val_shape_iou)
		self._log_val_stats()
		self._log_params()

	def _train_iter(self):
		self.optimizer.zero_grad()
		self.iter_timer.tic()
		data_time, batch_loss = 0, 0

		# Get training data
		for sub_iter in range(self.config.iter_size):
			self.data_timer.tic()
			shape_pairs, queries_field, target = self._fetch_data()
			data_time += self.data_timer.toc(False)

			# Feed forward
			soutput = self.model(*shape_pairs)
			out_field = soutput.interpolate(queries_field)
			# The output of the network is not sorted
			target = target.long().to(self.device)

			loss = self.criterion(out_field.F, target.long())

			# Compute and accumulate gradient
			loss /= self.config.iter_size
			batch_loss += loss.item()
			loss.backward()

		# Update number of steps
		self.optimizer.step()
		if self.config.scheduler != "ReduceLROnPlateau":
			self.scheduler.step()
		torch.cuda.empty_cache()

		self.data_time_avg.update(data_time)
		self.iter_time_avg.update(self.iter_timer.toc(False))

		pred = torch.max(out_field.F[:, 1:], 1)[1] + 1
		score = precision_at_one_partnet(pred, target)
		self.losses.update(batch_loss, target.size(0))
		self.scores.update(score, target.size(0))

	def _validate(self):
		loss, score, part_iou, shape_iou = Trainer.test(self.model, self.val_data_loader, self.data_loader, self.config.k_neighbors,
		                                        self.config, self.mink_settings)
		self.writer.add_scalar('validation/PartIoU', part_iou, self.curr_iter)
		self.writer.add_scalar('validation/ShapeIoU', shape_iou, self.curr_iter)
		self.writer.add_scalar('validation/loss', loss, self.curr_iter)
		self.writer.add_scalar('validation/precision_at_1', score, self.curr_iter)

		return loss, score, part_iou, shape_iou

	def _fetch_data(self):
		coords, input, target, neighbors = self.data_iter.next()
		queries_field = ME.TensorField(
			features=input,
			coordinates=coords,
			quantization_mode=self.mink_settings["q_mode"],
			minkowski_algorithm=self.mink_settings["mink_algo"],
			device=self.device)
		queries = queries_field.sparse()
		shape_pairs = [queries]
		if self.config.k_neighbors > 0:
			# Get neighbors
			query_neighbors = []
			query_neighbors_field = get_neighbors(self.data_loader.dataset, neighbors, self.config.k_neighbors)
			for q_idx, q_neighbors in enumerate(query_neighbors_field):
				query_neighbors_field[q_idx] = ME.TensorField(
					features=q_neighbors[0],
					coordinates=q_neighbors[1],
					quantization_mode=self.mink_settings["q_mode"],
					minkowski_algorithm=self.mink_settings["mink_algo"],
					device=self.device)
				query_neighbors.append(query_neighbors_field[q_idx].sparse())
			shape_pairs.append(query_neighbors)

		return shape_pairs, queries_field, target

	def _construct_shape_graph(self, recalculate=False):
		if recalculate:
			logging.info("===> Recalculate shape graph for training split")
		else:
			logging.info("===> Construct shape graph for training split")

		self.data_loader.dataset.neighbors = \
			construct_shape_graph(model=self.model, device=self.device, query_dataset=self.data_loader.dataset,
								  key_dataset=None, K=self.config.k_neighbors, writer=self.writer, tag='training',
								  step=self.n_graph_construction, random_pairs=False if recalculate else True,
								  mink_settings=self.mink_settings)

		if recalculate:
			logging.info("===> Recalculate shape graph for validation split")
		else:
			logging.info("===> Construct shape graph for validation split")
		self.val_data_loader.dataset.neighbors = \
			construct_shape_graph(model=self.model, device=self.device, query_dataset=self.val_data_loader.dataset,
								  key_dataset=self.data_loader.dataset, K=self.config.k_neighbors,
								  writer=self.writer, tag='validation', step=self.n_graph_construction,
								  random_pairs=False if recalculate else True, mink_settings=self.mink_settings)

	def _log_stats(self):
		lr = ', '.join(['{:.3e}'.format(self.optimizer.param_groups[0]['lr'])])
		debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t"\
			.format(self.epoch, self.curr_iter, self.data_len, self.losses.avg, lr)
		debug_str += "Score {:.3f}\tData time: {:.4f}, Total iter time: {:.4f}"\
			.format(self.scores.avg, self.data_time_avg.avg, self.iter_time_avg.avg)
		logging.info(debug_str)
		# Reset timers
		self.data_time_avg.reset()
		self.iter_time_avg.reset()
		# Write logs
		self.writer.add_scalar('training/loss', self.losses.avg, self.curr_iter)
		self.writer.add_scalar('training/precision_at_1', self.scores.avg, self.curr_iter)
		self.writer.add_scalar('training/learning_rate', self.optimizer.param_groups[0]['lr'], self.curr_iter)

	def _log_val_stats(self):
		logging.info("Current best Part IoU: {:.3f} at iter {}"
		             .format(self.best_val_part_iou, self.best_val_part_iou_iter))
		logging.info("Current best Shape IoU: {:.3f} at iter {}"
		             .format(self.best_val_shape_iou, self.best_val_shape_iou_iter))
		logging.info("Current best Loss: {:.3f} at iter {}"
		             .format(self.best_val_loss, self.best_val_loss_iter))
		logging.info("Current best Score: {:.3f} at iter {}"
		             .format(self.best_val_acc, self.best_val_acc_iter))

	def _log_params(self):
		for name, weight in self.model.named_parameters():
			self.writer.add_histogram(self.model_name+'/'+name, weight, self.epoch)
			if weight.grad is not None:
				self.writer.add_histogram(self.model_name + '/' + name + '.grad', weight.grad, self.epoch)

	def _save_curr_checkpoint(self, postfix=None):
		csn_data = None
		if self.config.k_neighbors > 0:
			csn_data = {"patience": self.patience,
			            "cooldown": self.cooldown,
			            "n_graph_construction": self.n_graph_construction,
			            "train_neighbors": self.data_loader.dataset.neighbors,
			            "val_neighbors": self.val_data_loader.dataset.neighbors}
		checkpoint(self.model, self.optimizer, self.epoch+1, self.curr_iter, self.config,
		           best_val_part_iou=self.best_val_part_iou, best_val_part_iou_iter=self.best_val_part_iou_iter,
		           best_val_shape_iou=self.best_val_shape_iou, best_val_shape_iou_iter=self.best_val_shape_iou_iter,
		           best_val_loss=self.best_val_loss,  best_val_loss_iter=self.best_val_loss_iter,
		           best_val_acc=self.best_val_acc, best_val_acc_iter=self.best_val_acc_iter, postfix=postfix,
		           csn_data=csn_data)

	def _save_best_checkpoints(self, val_loss, val_score, val_part_iou, val_shape_iou):
		if val_part_iou > self.best_val_part_iou:
			self.best_val_part_iou = val_part_iou
			self.best_val_part_iou_iter = self.curr_iter
			self._save_curr_checkpoint(postfix='best_part_iou')
		if val_shape_iou > self.best_val_shape_iou:
			self.best_val_shape_iou = val_shape_iou
			self.best_val_shape_iou_iter = self.curr_iter
			self._save_curr_checkpoint(postfix='best_shape_iou')
		if val_loss < self.best_val_loss:
			self.best_val_loss = val_loss
			self.best_val_loss_iter = self.curr_iter
			self._save_curr_checkpoint(postfix='best_loss')
		if val_score > self.best_val_acc:
			self.best_val_acc = val_score
			self.best_val_acc_iter = self.curr_iter
			self._save_curr_checkpoint(postfix='best_acc')

	def _resume(self):
		checkpoint_fn = self.config.resume + '/weights.pth'
		if osp.isfile(checkpoint_fn):
			logging.info("=> Loading checkpoint '{}'".format(checkpoint_fn))
			state = torch.load(checkpoint_fn)
			self.curr_iter = state['iteration'] + 1
			self.epoch = state['epoch']
			self.model.load_state_dict(state['state_dict'])
			logging.info("===> Model loaded")
			if self.config.resume_optimizer:
				self.scheduler = \
					initialize_scheduler(self.optimizer, self.config, last_step=self.curr_iter, factor=self.lr_factor)
				self.optimizer.load_state_dict(state['optimizer'])
				logging.info("===> Optimizer loaded")
			if 'csn_data' in state:
				csn_data = state['csn_data']
				self.patience = csn_data["patience"]
				self.cooldown = csn_data["cooldown"]
				self.n_graph_construction = csn_data["n_graph_construction"]
				logging.info("===> Patience={}, Cooldown={}, #Graph construction={}" .format(self.patience, self.cooldown,
				                                                                             self.n_graph_construction))
				self.data_loader.dataset.neighbors = csn_data["train_neighbors"]
				logging.info("===> Training neighbors loaded - {}".format(self.data_loader.dataset.neighbors))
				self.val_data_loader.dataset.neighbors = csn_data["val_neighbors"]
				logging.info("===> Validation neighbors loaded - {}".format(self.val_data_loader.dataset.neighbors))
			if 'best_val_part_iou' in state:
				self.best_val_part_iou = state['best_val_part_iou']
				self.best_val_part_iou_iter = state['best_val_part_iou_iter']
			if 'best_val_shape_iou' in state:
				self.best_val_shape_iou = state['best_val_shape_iou']
				self.best_val_shape_iou_iter = state['best_val_shape_iou_iter']
			if 'best_val_loss' in state:
				self.best_val_loss = state['best_val_loss']
				self.best_val_loss_iter = state['best_val_loss_iter']
			if 'best_val_acc' in state:
				self.best_val_acc = state['best_val_acc']
				self.best_val_acc_iter = state['best_val_acc_iter']
			logging.info("=> Loaded checkpoint '{}' (epoch {})".format(checkpoint_fn, state['epoch']))
		else:
			raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))

	@staticmethod
	def print_info(iteration, max_iteration, data_time, iter_time, losses, scores, part_iou, shape_iou):
		debug_str = "{}/{}: ".format(iteration + 1, max_iteration)
		debug_str += "Data time: {:.4f}, Iter time: {:.4f}".format(data_time, iter_time)
		debug_str += "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f})\t" \
		             "Score {top1.val:.3f} (AVG: {top1.avg:.3f})\t" \
		             "Part IoU {Part_IoU:.3f} Shape IoU {Shape_IoU:.3f}\n" \
			.format(loss=losses, top1=scores, Part_IoU=part_iou, Shape_IoU=shape_iou)

		logging.info(debug_str)

	@staticmethod
	def test(model, data_loader, train_data_loader, K, config, mink_settings):
		device = get_torch_device(config.is_cuda)
		dataset = data_loader.dataset
		num_labels = dataset.NUM_LABELS
		global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()
		criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)
		losses, scores, ious, shape_iou, part_iou = AverageMeter(), AverageMeter(), {}, 0.0, 0.0

		logging.info('===> Start testing')

		global_timer.tic()
		data_iter = data_loader.__iter__()
		max_iter = len(data_loader)
		max_iter_unique = max_iter

		# Fix batch normalization running mean and std
		model.eval()

		# Clear cache (when run in val mode, cleanup training cache)
		torch.cuda.empty_cache()

		if not config.is_train:
			save_pred_dir = config.save_pred_dir
			os.makedirs(save_pred_dir, exist_ok=True)
			if os.listdir(save_pred_dir):
				raise ValueError(f'Directory {save_pred_dir} not empty. '
				                 'Please remove the existing prediction.')

		with torch.no_grad():
			for iteration in range(max_iter):
				data_timer.tic()
				coords, input, target, neighbors = data_iter.next()

				queries_field = ME.TensorField(
					features=input,
					coordinates=coords,
					quantization_mode=mink_settings["q_mode"],
					minkowski_algorithm=mink_settings["mink_algo"],
					device=device)
				queries = queries_field.sparse()
				shape_pairs = [queries]
				if K > 0:
					# Get neighbors
					query_neighbors = []
					query_neighbors_field = get_neighbors(train_data_loader.dataset, neighbors, K)
					for q_idx, q_neighbors in enumerate(query_neighbors_field):
						query_neighbors_field[q_idx] = ME.TensorField(
							features=q_neighbors[0],
							coordinates=q_neighbors[1],
							quantization_mode=mink_settings["q_mode"],
							minkowski_algorithm=mink_settings["mink_algo"],
							device=device)
						query_neighbors.append(query_neighbors_field[q_idx].sparse())
					shape_pairs.append(query_neighbors)

				data_time = data_timer.toc(False)

				# Preprocess input
				iter_timer.tic()

				# Feed forward
				soutput = model(*shape_pairs)
				out_field = soutput.interpolate(queries_field)
				output = out_field.F

				pred = torch.max(output[:, 1:], 1)[1] + 1
				iter_time = iter_timer.toc(False)

				num_sample = target.shape[0]
				target = target.to(device)
				cross_ent = criterion(output, target.long())
				losses.update(float(cross_ent), num_sample)
				scores.update(precision_at_one_partnet(pred, target), num_sample)
				ious[iteration] = calculate_iou(ground=target.cpu().numpy(), prediction=pred.cpu().numpy(),
				                                num_labels=num_labels)

				if iteration % config.test_stat_freq == 0 and iteration > 0:
					shape_iou = calculate_shape_iou(ious=ious) * 100
					part_iou = calculate_part_iou(ious=ious, num_labels=num_labels) * 100
					Trainer.print_info(iteration, max_iter_unique, data_time, iter_time, losses, scores, part_iou, shape_iou)

				if iteration % config.empty_cache_freq == 0:
					# Clear cache
					torch.cuda.empty_cache()

		global_time = global_timer.toc(False)

		shape_iou = calculate_shape_iou(ious=ious) * 100
		part_iou = calculate_part_iou(ious=ious, num_labels=num_labels) * 100
		Trainer.print_info(iteration, max_iter_unique, data_time, iter_time, losses, scores, part_iou, shape_iou)

		if not config.is_train:
			# Log results
			buf = "Shape IoU: " + str(np.round(shape_iou, 2)) + "\nPart IoU: " + str(np.round(part_iou, 2))
			with open(os.path.join(save_pred_dir, "results_log.txt"), 'w') as fout_txt:
				fout_txt.write(buf)

		logging.info("Finished test. Elapsed time: {:.4f}".format(global_time))

		return losses.avg, scores.avg, part_iou, shape_iou
