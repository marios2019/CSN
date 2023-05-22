import numpy as np
import os
import sys
import json
import logging
from easydict import EasyDict as edict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import torch
from torch import nn

import MinkowskiEngine as ME

from lib.config import get_config
from lib.trainer_seg import Trainer
from lib.utils import get_torch_device, count_parameters
from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset
from models import load_model

ch = logging.StreamHandler(sys.stdout)
logging.basicConfig(format=os.uname()[1].split('.')[0] + ' -- %(asctime)s -- %(filename)s -- %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[ch])


def main():
    config, mink_settings = get_config()
    if config.resume:
        json_config = json.load(open(config.resume + '/config.json', 'r'))
        json_config['resume'] = config.resume
        config = edict(json_config)

    if config.is_cuda and not torch.cuda.is_available():
        raise Exception("No GPU found!!!")
    target_device = get_torch_device(config.is_cuda)

    logging.info('===> Configurations')
    dconfig = vars(config)
    for k in dconfig:
        logging.info('    {}: {}'.format(k, dconfig[k]))

    # Init seed
    torch.manual_seed(config.seed)

    DatasetClass = load_dataset(config.dataset)

    logging.info('===> Initializing dataloader')
    if config.is_train:
        batch_size = config.batch_size
        train_data_loader = initialize_data_loader(
            DatasetClass,
            config,
            phase=config.train_phase,
            num_workers=config.num_workers,
            shift=config.shift,
            jitter=config.jitter,
            rot_aug=config.rot_aug,
            scale=config.scale,
            shuffle=True,
            repeat=True,
            batch_size=batch_size,
            limit_numpoints=config.train_limit_numpoints)
        val_data_loader = initialize_data_loader(
            DatasetClass,
            config,
            num_workers=config.num_val_workers,
            phase=config.val_phase,
            shift=False,
            jitter=False,
            rot_aug=False,
            scale=False,
            shuffle=True,
            repeat=False,
            batch_size=config.val_batch_size,
            limit_numpoints=False)
        if train_data_loader.dataset.NUM_IN_CHANNEL is not None:
            num_in_channel = train_data_loader.dataset.NUM_IN_CHANNEL
        else:
            num_in_channel = 3
        num_labels = train_data_loader.dataset.NUM_LABELS
    else:
        test_data_loader = initialize_data_loader(
            DatasetClass,
            config,
            num_workers=config.num_workers,
            phase=config.test_phase,
            shift=False,
            jitter=False,
            rot_aug=False,
            scale=False,
            shuffle=False,
            repeat=False,
            batch_size=config.test_batch_size,
            limit_numpoints=False)
        if test_data_loader.dataset.NUM_IN_CHANNEL is not None:
            num_in_channel = test_data_loader.dataset.NUM_IN_CHANNEL
        else:
            num_in_channel = 3
        num_labels = test_data_loader.dataset.NUM_LABELS

    logging.info('===> Building model')
    NetClass = load_model(config.model)
    model = NetClass(num_in_channel, num_labels, config)
    logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__, count_parameters(model)))

    logging.info(model)
    model = model.to(target_device)

    test_iter = 0
    # Load weights if specified by the parameter.
    if config.weights.lower() != 'none':
        logging.info('===> Loading weights: ' + config.weights)
        state = torch.load(config.weights)
        model.load_state_dict(state['state_dict'])
        if 'iteration' in state:
            test_iter = state['iteration']

    if config.is_train:
        trainer = Trainer(model=model, data_loader=train_data_loader, val_data_loader=val_data_loader, config=config,
                          mink_settings=mink_settings)
        trainer.train()
    else:
        loss, score, part_iou, shape_iou = Trainer.test(model=model, data_loader=test_data_loader, config=config,
                                                        mink_settings=mink_settings)
        logging.info("Test split Part IOU: {:.3f} at iter {}".format(part_iou, test_iter))
        logging.info("Test split Shape IOU: {:.3f} at iter {}".format(shape_iou, test_iter))
        logging.info("Test split Loss: {:.3f} at iter {}".format(loss, test_iter))
        logging.info("Test Score: {:.3f} at iter {}".format(score, test_iter))


if __name__ == '__main__':
    main()
