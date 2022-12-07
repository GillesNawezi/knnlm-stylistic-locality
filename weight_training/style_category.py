import numpy as np
import torch
import torch.nn as nn
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pathlib
from fairseq import utils
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

num_retrieved = 1024
dataset = "style_category_wiki_fine_tune" #"style_category_dataset"
global_path = str(pathlib.Path(__file__).parent.parent.resolve()) + "/"
n=50000
epochs = 1000

context_vecs = np.load(global_path + f'saved_tensors/{dataset}/valid_context_cache.npy')
dists = np.load(f'saved_tensors/{dataset}/valid_proj_dist_cache.npy').reshape(-1, num_retrieved)
pkg_locality = np.load(f'saved_tensors/{dataset}/valid_pkg_locality_cache.npy').reshape(-1, num_retrieved)
proj_locality = np.load(f'saved_tensors/{dataset}/valid_proj_locality_cache.npy').reshape(-1, num_retrieved)
index_masks = np.load(f'saved_tensors/{dataset}/valid_proj_index_mask_cache.npy').reshape(-1, num_retrieved)
lm_probs = np.load(f'saved_tensors/{dataset}/valid_lm_prob_cache.npy')

print(context_vecs[0])

print(context_vecs.shape)
print(dists.shape)
print(pkg_locality.shape)
print(proj_locality.shape)
print(index_masks.shape)
print(lm_probs.shape)

"""
test_context_vecs = np.load(f'saved_tensors/{dataset}/test_context_cache.npy')
test_dists = np.load(f'saved_tensors/{dataset}/test_proj_dist_cache.npy').reshape(-1, num_retrieved)
test_pkg_locality = np.load(f'saved_tensors/{dataset}/test_pkg_locality_cache.npy').reshape(-1, num_retrieved)
test_proj_locality = np.load(f'saved_tensors/{dataset}/test_proj_locality_cache.npy').reshape(-1, num_retrieved)
test_index_masks = np.load(f'saved_tensors/{dataset}/test_proj_index_mask_cache.npy').reshape(-1, num_retrieved)
test_lm_probs = np.load(f'saved_tensors/{dataset}/test_lm_prob_cache.npy')
"""

