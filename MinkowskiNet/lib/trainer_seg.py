import logging
import os.path as osp
import os
import numpy as np

import torch
from torch import nn

from tensorboardX import SummaryWriter

from lib.utils import checkpoint, precision_at_one_partnet, calculate_iou, calculate_shape_iou, calculate_part_iou, \
	Timer, AverageMeter, get_torch_device
from lib.solvers import initialize_optimizer, initialize_scheduler

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

		self.optimizer = initialize_optimizer(self.model.parameters(), self.config)
		self.lr_factor = 0.5
		self.scheduler = initialize_scheduler(self.optimizer, self.config, factor=self.lr_factor, verbose=True)
		self.criterion = nn.CrossEntropyLoss(ignore_index=self.config.ignore_label)

		self.best_val_part_iou, self.best_val_part_iou_iter = 0, 0
		self.best_val_shape_iou, self.best_val_shape_iou_iter = 0, 0
		self.best_val_loss, self.best_val_loss_iter = np.Inf, 0
		self.best_val_acc, self.best_val_acc_iter = 0, 0
		self.curr_iter, self.epoch, self.is_training = 1, 1, True

	def train(self):
		# Train the network
		self.model.train()
		logging.info('===> Start training')

		if self.config.resume:
			# Resume training
			self._resume()

		if self.config.save_param_histogram:
			# Log initial params histograms
			self._log_params()

		self.data_iter = self.data_loader.__iter__()
		torch.autograd.set_detect_anomaly(True)

		# Number of batches
		self.data_len = len(self.data_loader)

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
			val_loss, val_score, val_part_iou, val_shape_iou = self._validate()
			self._save_best_checkpoints(val_loss, val_score, val_part_iou, val_shape_iou)
			self._log_val_stats()

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

		# Get training data
		self.data_timer.tic()
		in_field, target = self._fetch_data()
		sinput = in_field.sparse()
		data_time = self.data_timer.toc(False)

		# Feed forward
		soutput = self.model(sinput)
		out_field = soutput.interpolate(in_field)
		target = target.long().to(self.device)

		loss = self.criterion(out_field.F, target.long())

		# Compute gradients
		batch_loss = loss.item()
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
		loss, score, part_iou, shape_iou = Trainer.test(self.model, self.val_data_loader, self.config, self.mink_settings)
		self.writer.add_scalar('validation/PartIoU', part_iou, self.curr_iter)
		self.writer.add_scalar('validation/ShapeIoU', shape_iou, self.curr_iter)
		self.writer.add_scalar('validation/loss', loss, self.curr_iter)
		self.writer.add_scalar('validation/precision_at_1', score, self.curr_iter)

		return loss, score, part_iou, shape_iou

	def _fetch_data(self):
		coords, input, target = self.data_iter.next()
		in_field = ME.TensorField(
			features=input,
			coordinates=coords,
			quantization_mode=self.mink_settings["q_mode"],
			minkowski_algorithm=self.mink_settings["mink_algo"],
			device=self.device)

		return in_field, target

	def _log_stats(self):
		lr = ', '.join(['{:.3e}'.format(self.optimizer.param_groups[0]['lr'])])
		debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t" \
			.format(self.epoch, self.curr_iter, self.data_len, self.losses.avg, lr)
		debug_str += "Score {:.3f}\tData time: {:.4f}, Total iter time: {:.4f}" \
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
		checkpoint(self.model, self.optimizer, self.epoch+1, self.curr_iter, self.config,
				   best_val_part_iou=self.best_val_part_iou, best_val_part_iou_iter=self.best_val_part_iou_iter,
				   best_val_shape_iou=self.best_val_shape_iou, best_val_shape_iou_iter=self.best_val_shape_iou_iter,
				   best_val_loss=self.best_val_loss,  best_val_loss_iter=self.best_val_loss_iter,
				   best_val_acc=self.best_val_acc, best_val_acc_iter=self.best_val_acc_iter, postfix=postfix)

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
			logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
			state = torch.load(checkpoint_fn)
			self.curr_iter = state['iteration'] + 1
			self.epoch = state['epoch']
			self.model.load_state_dict(state['state_dict'])
			if self.config.resume_optimizer:
				self.scheduler = \
					initialize_scheduler(self.optimizer, self.config, last_step=self.curr_iter, factor=self.lr_factor)
				self.optimizer.load_state_dict(state['optimizer'])
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
			logging.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_fn, state['epoch']))
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
	def test(model, data_loader, config, mink_settings):
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
				raise ValueError(f'Directory {save_pred_dir} not empty. Please remove the existing prediction.')

		with torch.no_grad():
			for iteration in range(max_iter):
				data_timer.tic()
				coords, input, target = data_iter.next()

				in_field = ME.TensorField(
					features=input,
					coordinates=coords,
					quantization_mode=mink_settings["q_mode"],
					minkowski_algorithm=mink_settings["mink_algo"],
					device=device)
				sinput = in_field.sparse()
				data_time = data_timer.toc(False)

				# Preprocess input
				iter_timer.tic()

				# Feed forward
				soutput = model(sinput)
				out_field = soutput.interpolate(in_field)
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
