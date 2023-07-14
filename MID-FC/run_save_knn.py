import os, sys
import csv
import math
import argparse
import pandas as pd

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

parser = argparse.ArgumentParser()
parser.add_argument('--ssa_logs_dir', type=str, default='logs/ssa_n_heads_1')

parser.add_argument('--num_workers', type=int, default=3)
parser.add_argument('--testing', action='store_true')
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--attention_type', type=str, default='ssa')
parser.add_argument('--K', type=int, default=10)

parser.add_argument('--n_heads', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gradient_accumulation_steps', type=int, default=2)

parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=16)

parser.add_argument('--cmd', action='store_true')
parser.add_argument('--job', action='store_true')

parser.add_argument('--gpu', type=str, default='1080ti')

args = parser.parse_args()

script = 'python save_knn_graph.py'

for k in range(0, len(names)):
    ratio, name = ratios[-1], names[k]

    if k < args.start or k > args.end:
      continue

    max_iter = int(max_iters[k] * ratio * muls[-1])
    num_classes = seg_num[k]

    cmds = [
        script,
        f'--ssa_logs_dir={args.ssa_logs_dir}/{name}',
        f'--graphs_dir={args.ssa_logs_dir}/knn_graphs/{name}',
        f'--partname={name}',
        f'--n_heads={args.n_heads}',
        f'--num_workers={args.num_workers}',
        f'--batch_size={args.batch_size}',
        f'--num_classes={num_classes}',
    ]
    
    if args.testing:
        cmds += [f'--testing']

    cmd = ' '.join(cmds)

    if args.cmd:
      print('\n', cmd, '\n')
      os.system(cmd)
      sys.exit()

    if args.job:
      sbatch = '#!/bin/sh\n'
      sbatch += f'#SBATCH --job-name={name}\n'
      sbatch += f'#SBATCH -o /work/siddhantgarg_umass_edu/int-vis/slurm/slurm-%j.out  # %j = job ID\n'
      sbatch += f'#SBATCH --partition=gypsum-{args.gpu}\n'
      sbatch += f'#SBATCH --gres=gpu:1\n'
      sbatch += f'#SBATCH -c 10\n'
      sbatch += f'#SBATCH --mem 30GB\n' 

      full_cmd = sbatch + '\n' + cmd 
      with open('run_partnet_train.sh', 'w') as f:
          f.write(full_cmd)

      print(full_cmd)
      os.system('sbatch run_partnet_train.sh')
