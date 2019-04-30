import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import networkx as nx
import datetime

import glob

parser = argparse.ArgumentParser('summarize the results in N file.results')
parser.add_argument('--dir', type=str)
parser.add_argument('--type', type=str)

args = parser.parse_args()

v_abs_error = []
v_rel_error = []
for filename in glob.glob(args.dir + r'/*.' + args.type):
    results_dict = torch.load(filename)
    v_iter = results_dict['v_iter']
    abs_error = results_dict['abs_error']
    rel_error = results_dict['rel_error']
    v_abs_error.append(abs_error[-1])
    v_rel_error.append(rel_error[-1])

v_abs_error = np.array(v_abs_error)
v_rel_error = np.array(v_rel_error)

print('abs_error:')
print('{} \pm {}'.format(v_abs_error.mean(), v_abs_error.std()))
print('{:.3f} \pm {:.3f}'.format(v_abs_error.mean(), v_abs_error.std()))
print('rel_error:')
print('{} \pm {}'.format(v_rel_error.mean() , v_rel_error.std()))
print('{:.1f} \pm {:.1f} %'.format(v_rel_error.mean() * 100, v_rel_error.std() *100))


# --dir results/mutualistic/grid  --type grid_our