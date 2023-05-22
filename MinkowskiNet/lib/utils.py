import json
import logging
import os
import errno
import time

import numpy as np
import torch


def checkpoint(model, optimizer, epoch, iteration, config, best_val_part_iou=None, best_val_part_iou_iter=None,
               best_val_shape_iou=None, best_val_shape_iou_iter=None, best_val_instance_iou=None,
               best_val_instance_iou_iter=None, best_val_category_iou=None, best_val_category_iou_iter=None,
               best_val_loss=None,  best_val_loss_iter=None, best_val_acc=None, best_val_acc_iter=None, postfix=None,
               csn_data=None):
    mkdir_p(config.log_dir)
    if config.overwrite_weights:
        if postfix is not None:
            filename = f"checkpoint_{config.model}{postfix}.pth"
        else:
            filename = f"checkpoint_{config.model}.pth"
    else:
        filename = f"checkpoint_{config.model}_iter_{iteration}.pth"
    checkpoint_file = config.log_dir + '/' + filename
    state = {
        'iteration': iteration,
        'epoch': epoch,
        'arch': config.model,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if csn_data is not None:
        state['csn_data'] = csn_data
    if best_val_part_iou is not None:
        state['best_val_part_iou'] = best_val_part_iou
        state['best_val_part_iou_iter'] = best_val_part_iou_iter
    if best_val_shape_iou is not None:
        state['best_val_shape_iou'] = best_val_shape_iou
        state['best_val_shape_iou_iter'] = best_val_shape_iou_iter
    if best_val_instance_iou is not None:
        state['best_val_instance_iou'] = best_val_instance_iou
        state['best_val_instance_iou_iter'] = best_val_instance_iou_iter
    if best_val_category_iou is not None:
        state['best_val_category_iou'] = best_val_category_iou
        state['best_val_category_iou_iter'] = best_val_category_iou_iter
    if best_val_loss  is not None:
        state['best_val_loss'] = best_val_loss
        state['best_val_loss_iter'] = best_val_loss_iter
    if best_val_acc is not None:
        state['best_val_acc'] = best_val_acc
        state['best_val_acc_iter'] = best_val_acc_iter
    json.dump(vars(config), open(config.log_dir + '/config.json', 'w'), indent=4)
    torch.save(state, checkpoint_file)
    logging.info(f"Checkpoint saved to {checkpoint_file}")

    if postfix is None:
        # Delete symlink if it exists
        if os.path.exists(f'{config.log_dir}/weights.pth'):
            os.remove(f'{config.log_dir}/weights.pth')
        # Create symlink
        os.system(f'cd {config.log_dir}; ln -s {filename} weights.pth')


def precision_at_one_partnet(pred, target, ignore_label=255):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target) | target.eq(0)
    correct = correct[target != ignore_label]
    correct = correct.view(-1)
    if correct.nelement():
        return correct.float().sum(0).mul(100.0 / correct.size(0)).item()
    else:
        return float('nan')


def calculate_iou(ground, prediction, num_labels):
    """
      Calculate point IOU
    :param ground: N x 1, numpy.ndarray(int)
    :param prediction: N x 1, numpy.ndarray(int)
    :param num_labels: int
    :return:
        metrics: dict: {
                        "label_iou": dict{label: iou (float)},
                        "intersection": dict{label: intersection (float)},
                        "union": dict{label: union (float)
                       }
    """

    label_iou, intersection, union = {}, {}, {}
    # Ignore undetermined
    prediction = np.copy(prediction)
    prediction[ground == 0] = 0

    for i in range(1, num_labels):
        # Calculate intersection and union for ground truth and predicted labels
        intersection_i = np.sum((ground == i) & (prediction== i))
        union_i = np.sum((ground == i) | (prediction == i))

        # If label i is present either on the gt or the pred set
        if union_i > 0:
            intersection[i] = float(intersection_i)
            union[i] = float(union_i)
            label_iou[i] = intersection[i] / union[i]

    metrics = {"label_iou": label_iou, "intersection": intersection, "union": union}

    return metrics


def calculate_shape_iou(ious):
    """
      Average label IOU and calculate overall shape IOU
    :param ious: dict: {
                        <model_name> : {
                                        "label_iou": dict{label: iou (float)},
                                        "intersection": dict{label: intersection (float)},
                                        "union": dict{label: union (float)
                                       }
                       }
    :return:
        avg_shape_iou: float
    """

    shape_iou, shape_iou_cnt = {}, 0

    for model_name, metrics in ious.items():
        # Average label iou per shape
        L_s = len(metrics["label_iou"])
        if L_s > 0:
            shape_iou[model_name] = np.nan_to_num(np.sum([v for v in metrics["label_iou"].values()]) / float(L_s))
            shape_iou_cnt += 1

    # Dataset avg shape iou
    avg_shape_iou = np.sum([v for v in shape_iou.values()]) / float(shape_iou_cnt)

    return avg_shape_iou


def calculate_part_iou(ious, num_labels):
    """
      Average intersection/union and calculate overall part IOU
    :param ious: dict: {
                        <model_name> : {
                                        "label_iou": dict{label: iou (float)},
                                        "intersection": dict{label: intersection (float)},
                                        "union": dict{label: union (float)
                                       }
                       }
    :param num_labels: int
    :return:
      avg_part_iou: float
    """

    intersection = {i: 0.0 for i in range(1, num_labels)}
    union = {i: 0.0 for i in range(1, num_labels)}

    for model_name, metrics in ious.items():
        for label in metrics["intersection"].keys():
            # Accumulate intersection and union for each label across all shapes
            intersection[label] += metrics["intersection"][label]
            union[label] += metrics["union"][label]

    # Calculate part IOU for each label
    part_iou = {}
    for key in range(1, num_labels):
        try:
            part_iou[key] = intersection[key] / union[key]
        except ZeroDivisionError:
            part_iou[key] = 0.0
    # Avg part IOU
    avg_part_iou = np.sum([v for v in part_iou.values()]) / float(num_labels - 1)

    return avg_part_iou


class WithTimer(object):
    """Timer for with statement."""

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        out_str = 'Elapsed: %s' % (time.time() - self.tstart)
        if self.name:
            logging.info('[{self.name}]')
        logging.info(out_str)


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.averate_time = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_txt(path):
    """Read txt file into lines.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_torch_device(is_cuda):
    return torch.device('cuda' if is_cuda else 'cpu')


def set_grad(model, flag=False):
    """
      Set requires_grad=flag for all parameters of the given model
    """

    for param in model.parameters():
        param.requires_grad = flag


def features_at(sparse_tensor, idx):
    """
      Get features from SparseTensor with batch_idx = idx
    """

    return sparse_tensor.F[sparse_tensor.C[:, 0] == idx]
