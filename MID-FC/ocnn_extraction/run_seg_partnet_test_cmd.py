import os, sys
import csv
import math
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, required=False,
                    default='0811_partnet_randinit_attn')
parser.add_argument('--gpu', type=int, required=False, default=0)
parser.add_argument('--mode', type=str, required=False, default='randinit')
parser.add_argument('--ckpt', type=str, required=False,
                    default='dataset/midnet_data/mid_d6_o6/model/iter_800000.ckpt')
parser.add_argument('--c', type=int, required=False, default=0)

parser.add_argument('--phase', type=str, required=True, default='train')
parser.add_argument('--input_path', type=str, required=True, default='./')
parser.add_argument('--logs_dir', type=str, required=True, default='data')


args = parser.parse_args()
alias = args.alias
gpu = args.gpu
mode = args.mode

factor = 2
batch_size = 32
ckpt = args.ckpt if mode != 'randinit' else '\'\''
module = 'run_seg_partnet_finetune.py' if mode != 'randinit' else 'run_seg_partnet.py'
script = 'python %s --config configs/seg_hrnet_partnet_pts.yaml' % module
if mode != 'randinit': script += ' SOLVER.mode %s ' % mode
data = 'dataset/partnet_segmentation/dataset'


names     = ['Bed', 'Bottle', 'Chair', 'Clock', 'Dishwasher', 'Display', 'Door', 
             'Earphone', 'Faucet', 'Knife', 'Lamp', 'Microwave', 'Refrigerator',
             'StorageFurniture', 'Table', 'TrashCan', 'Vase']
train_num = [ 133,   315,  4489,  406,   111,   633,   149,  147,  
              435,   221,  1554,  133,   136,  1588,  5707,  221,  741]
max_iters = [3000,  3000, 20000, 5000,  3000,  5000,  3000, 3000, 
             5000,  3000, 10000, 3000,  3000, 10000, 20000, 3000, 10000]
test_iters= [ 100,   100,   800,  400,   200,   400,   200,  200, 
              400,   200,   800,  200,   200,   800,   800,  200,   800]
test_num  = [  37,    84,  1217,   98,    51,   191,    51,   53, 
              132,    77,   419,   39,    31,   451,  1668,   63,   233]
val_num   = [  24,    37,   617,   50,    19,   104,    25,   28,   
               81,    29,   234,   12,    20,   230,   843,   37,   102]
seg_num   = [  15,     9,    39,   11,     7,     4,     5,   10,  
               12,    10,    41,    6,     7,    24,    51,   11,     6]
ratios    = [0.01,  0.02,  0.05, 0.10,  0.20,  0.50, 1.00]
muls      = [   2,     2,     2,    1,     1,     1,    1]  # longer iter when data < 10%


for i in range(len(ratios)-1, -1, -1):
  for k in range(len(names)-1, -1, -1):
    ratio, name = ratios[i], names[k]
    max_iter = int(max_iters[k] * ratio * muls[i])
    step_size1, step_size2 = int(0.5 * max_iter), int(0.25 * max_iter)
    test_every_iter = int(test_iters[k] * ratio * muls[i])
    take = int(math.ceil(train_num[k] * ratio))

    ckpt_dir = 'logs/partnet_weights_and_logs/partnet_finetune/{}/ratio_{:.2f}/model'.format(name, ratio)
    ckpt_files = os.listdir(ckpt_dir)
    for fname in ckpt_files:
      if 'ckpt.index' in fname:
        fname = fname.split('.')[0]
        break
    ckpt = 'logs/partnet_weights_and_logs/partnet_finetune/{}/ratio_{:.2f}/model/{}.ckpt'.format(name, ratio, fname)

    if args.phase == 'train':
      input_path = '{}/{}_train_level3.tfrecords'.format(data, name)'
      logs_dir = os.path.join(args.logs_dir, name, 'train_features')
    else:
      input_path = '{}/{}_test_level3.tfrecords'.format(data, name)'
      logs_dir = os.path.join(args.logs_dir, name, 'test_features')

    cmds = [
        script,
        'SOLVER.gpu {},'.format(gpu),
        'SOLVER.logdir {}'.format(logs_dir),
        'SOLVER.max_iter {}'.format(max_iter),
        'SOLVER.step_size {},{}'.format(step_size1, step_size2),
        'SOLVER.test_every_iter {}'.format(test_every_iter),
        'SOLVER.test_iter {}'.format(train_num[k]),
        'SOLVER.ckpt {}'.format(ckpt),
        'SOLVER.run test',
        'DATA.train.location {}/{}_train_level3.tfrecords'.format(data, name),
        'DATA.train.take {}'.format(take),
        'DATA.test.location {}'.format(input_path),
        'MODEL.nout {}'.format(seg_num[k]),
        'MODEL.factor {}'.format(factor),
        'LOSS.num_class {}'.format(seg_num[k])]

    '''
    modify below code to excute the cmd = ' '.join(cmds) to submit the cmd for individual shapes
    '''

    sbatch = '#!/bin/sh\n'
    sbatch += f'#SBATCH --job-name={name}\n'
    sbatch += f'#SBATCH -o /work/siddhantgarg_umass_edu/int-vis/slurm/slurm-%j.out  # %j = job ID\n'
    sbatch += f'#SBATCH --partition=gypsum-1080ti\n'
    sbatch += f'#SBATCH --gres=gpu:1\n'
    sbatch += f'#SBATCH -c 10\n'
    sbatch += f'#SBATCH --mem 20GB\n' 

    full_cmd = sbatch + '\n' + cmd 
    with open('run_partnet_train.sh', 'w') as f:
      f.write(full_cmd)
    print(full_cmd)
    os.system('sbatch run_partnet_train.sh')
  break
