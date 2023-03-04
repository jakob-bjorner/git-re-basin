import argparse
import pickle
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
from flax.serialization import from_bytes
from jax import random
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from cifar100_resnet20_train import make_stuff
from datasets import load_cifar100
from resnet20 import BLOCKS_PER_GROUP, ResNet
from utils import (ec2_get_instance_type, flatten_params, lerp, timeblock, unflatten_params)
from weight_matching import (apply_permutation, resnet20_permutation_spec, weight_matching)

import os
import pickle

from collections import defaultdict
from typing import NamedTuple

import jax.numpy as jnp
from jax import random
import pdb

from utils import rngmix

import clip
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast

class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict

# def mlp_permutation_spec(num_hidden_layers: int) -> PermutationSpec:
#   """We assume that one permutation cannot appear in two axes of the same weight array."""
#   assert num_hidden_layers >= 1
#   return PermutationSpec(
#       perm_to_axes={
#           f"P_{i}": [(f"Dense_{i}/kernel", 1), (f"Dense_{i}/bias", 0), (f"Dense_{i+1}/kernel", 0)]
#           for i in range(num_hidden_layers)
#       },
#       axes_to_perm={
#           "Dense_0/kernel": (None, "P_0"),
#           **{f"Dense_{i}/kernel": (f"P_{i-1}", f"P_{i}")
#              for i in range(1, num_hidden_layers)},
#           **{f"Dense_{i}/bias": (f"P_{i}", )
#              for i in range(num_hidden_layers)},
#           f"Dense_{num_hidden_layers}/kernel": (f"P_{num_hidden_layers-1}", None),
#           f"Dense_{num_hidden_layers}/bias": (None, ),
#       })

def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)

def resnet20_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {f"{name}/kernel": (None, None, p_in, p_out)}
    norm = lambda name, p: {
        f"{name}/scale": (p, ), 
        f"{name}/bias": (p, ), 
        f"{name}/mean": (p, ), 
        f"{name}/var": (p, )
    }
    dense = lambda name, p_in, p_out: {f"{name}/kernel": (p_in, p_out), f"{name}/bias": (p_out, )}

    # This is for easy blocks that use a residual connection, without any change in the number of channels.
    easyblock = lambda name, p: {
      **conv(f"{name}/conv1", p, f"P_{name}_inner"),
      **norm(f"{name}/norm1", f"P_{name}_inner"),
      **conv(f"{name}/conv2", f"P_{name}_inner", p),
      **norm(f"{name}/norm2", p)
    }

    # This is for blocks that use a residual connection, but change the number of channels via a Conv.
    shortcutblock = lambda name, p_in, p_out: {
      **conv(f"{name}/conv1", p_in, f"P_{name}_inner"),
      **norm(f"{name}/norm1", f"P_{name}_inner"),
      **conv(f"{name}/conv2", f"P_{name}_inner", p_out),
      **norm(f"{name}/norm2", p_out),
      **conv(f"{name}/shortcut/layers_0", p_in, p_out),
      **norm(f"{name}/shortcut/layers_1", p_out),
    }

    return permutation_spec_from_axes_to_perm({
      **conv("conv1", None, "P_bg0"),
      **norm("norm1", "P_bg0"),
      #
      **easyblock("blockgroups_0/blocks_0", "P_bg0"),
      **easyblock("blockgroups_0/blocks_1", "P_bg0"),
      **easyblock("blockgroups_0/blocks_2", "P_bg0"),
      #
      **shortcutblock("blockgroups_1/blocks_0", "P_bg0", "P_bg1"),
      **easyblock("blockgroups_1/blocks_1", "P_bg1"),
      **easyblock("blockgroups_1/blocks_2", "P_bg1"),
      #
      **shortcutblock("blockgroups_2/blocks_0", "P_bg1", "P_bg2"),
      **easyblock("blockgroups_2/blocks_1", "P_bg2"),
      **easyblock("blockgroups_2/blocks_2", "P_bg2"),
      #
      **dense("dense", "P_bg2", None),
    })
    
def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
    # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            w = jnp.take(w, perm[p], axis=axis)

    return w

def apply_permutation(ps: PermutationSpec, perm, params):
  """Apply a `perm` to `params`."""
  return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}

def weight_matching(rng,
                    ps: PermutationSpec,
                    params_a,
                    params_b,
                    max_iter=100,
                    init_perm=None,
                    silent=False):
  """Find a permutation of `params_b` to make them match `params_a`."""
  perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}

  perm = {p: jnp.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
  perm_names = list(perm.keys())
  
  for iteration in tqdm(range(max_iter)):
    progress = False
    for p_ix in random.permutation(rngmix(rng, iteration), len(perm_names)):
      p = perm_names[p_ix]
      n = perm_sizes[p]
      A = jnp.zeros((n, n))
      for wk, axis in ps.perm_to_axes[p]:
        # pdb.set_trace()
        try:
          w_a = params_a[wk]
        except:
          pdb.set_trace()
        w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
        w_a = jnp.moveaxis(w_a, axis, 0).reshape((n, -1))
        w_b = jnp.moveaxis(w_b, axis, 0).reshape((n, -1))
        A += w_a @ w_b.T

      ri, ci = linear_sum_assignment(A, maximize=True)
      assert (ri == jnp.arange(len(ri))).all()

      oldL = jnp.vdot(A, jnp.eye(n)[perm[p]])
      newL = jnp.vdot(A, jnp.eye(n)[ci, :])
      if not silent: print(f"{iteration}/{p}: {newL - oldL}")
      progress = progress or newL > oldL + 1e-12

      perm[p] = jnp.array(ci)

    if not progress:
      break

  return perm

def load_pickle(path):
    return pickle.load(open(path, 'rb'))


def find_pairs(str_splits):
    pairs = []
    for i, str_split_i in enumerate(str_splits):
        split_i = set([int(k) for k in str_split_i.split('_')])
        for str_split_j in str_splits[i+1:]:
            split_j = set([int(k) for k in str_split_j.split('_')])
            if len(split_i.intersection(split_j)) == 0:
                pairs.append((str_split_i, str_split_j))
    return pairs


def split_str_to_ints(split):
    return [int(i) for i in split.split('_')]


def is_valid_pair(model_dir, pair, model_type):
    paths = os.listdir(os.path.join(model_dir, pair[0]))
    for path in paths:
        if model_type in path:
            return True
    return False


model_dir = '/srv/share2/gstoica3/checkpoints/cifar50_traincliphead/'
model_name = 'resnet20x4'
eval_pair = 0
pair = [pair for pair in find_pairs(os.listdir(model_dir)) if is_valid_pair(model_dir, pair, model_name)][eval_pair]
model_save_paths = [os.path.join(model_dir, split, f'{model_name}_flax_v0.pkl') for split in pair]


a_sd = load_pickle(os.path.join(model_save_paths[0]))

b_sd = load_pickle(os.path.join(model_save_paths[1]))

a_sd_union = flatten_params(a_sd['params'])
a_sd_union.update(flatten_params(a_sd['batch_stats']))

b_sd_union = flatten_params(b_sd['params'])
b_sd_union.update(flatten_params(b_sd['batch_stats']))

permutation_spec = resnet20_permutation_spec()

final_permutation = weight_matching(
    random.PRNGKey(0), permutation_spec,
    # flatten_params(model_a), 
    # flatten_params(model_b)
    a_sd_union, 
    b_sd_union
)

model_b_params_clever = unflatten_params(apply_permutation(permutation_spec, final_permutation, flatten_params(b_sd['params'])))
model_b_stats_clever = unflatten_params(apply_permutation(permutation_spec, final_permutation, flatten_params(b_sd['batch_stats'])))
clever_params_sd = lerp(.5, a_sd['params'], model_b_params_clever.unfreeze())
clever_stats_sd = lerp(.5, a_sd['batch_stats'], model_b_stats_clever.unfreeze())
clever_p_sd = {'params': clever_params_sd, 'batch_stats': clever_stats_sd}
train_ds, test_ds = load_cifar100()

model = ResNet(blocks_per_group=BLOCKS_PER_GROUP["resnet20"], num_classes=512, width_multiplier=4)
stuff = make_stuff(model)
test_loss, test_acc1, test_acc5 = stuff["dataset_loss_and_accuracies"](clever_p_sd, test_ds, 1000)
print('Acc: {}'.format(test_acc1))
save_path = os.path.join(model_dir, pair[0], 'flax_cifar50_2_permuted_to_1.pkl')
print(save_path)
pickle.dump(clever_p_sd, open(save_path, 'wb'))