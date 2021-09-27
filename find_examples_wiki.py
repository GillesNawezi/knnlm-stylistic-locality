import numpy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from fairseq import utils


def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = np.log(1 - coeff)
    coeffs[1] = np.log(coeff)
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

    return curr_prob


num_retrieved = 1024

# context_vecs = np.load('saved_tensors/wikitext-103/valid_context_cache.npy')
# dists = np.load('saved_tensors/wikitext-103/valid_proj_dist_cache.npy').reshape(-1, num_retrieved)
# pkg_locality = np.load('saved_tensors/wikitext-103/valid_pkg_locality_cache.npy').reshape(-1, num_retrieved)
# proj_locality = np.load('saved_tensors/wikitext-103/valid_proj_locality_cache.npy').reshape(-1, num_retrieved)
# index_masks = np.load('saved_tensors/wikitext-103/valid_proj_index_mask_cache.npy').reshape(-1, num_retrieved)
# lm_probs = np.load('saved_tensors/wikitext-103/valid_lm_prob_cache.npy')
#
# context_vecs = torch.from_numpy(context_vecs).float()
# dists = torch.from_numpy(dists).float()
# pkg_locality = torch.from_numpy(pkg_locality)
# proj_locality = torch.from_numpy(proj_locality)
# index_masks = torch.from_numpy(index_masks)

test_context_vecs = np.load('saved_tensors/wikitext-103/test_context_cache.npy')
test_dists = np.load('saved_tensors/wikitext-103/test_proj_dist_cache.npy').reshape(-1, num_retrieved)
test_knns = np.load('saved_tensors/wikitext-103/test_knn_cache.npy').reshape(-1, num_retrieved)
test_pkg_locality = np.load('saved_tensors/wikitext-103/test_pkg_locality_cache.npy').reshape(-1, num_retrieved)
test_proj_locality = np.load('saved_tensors/wikitext-103/test_proj_locality_cache.npy').reshape(-1, num_retrieved)
test_index_masks = np.load('saved_tensors/wikitext-103/test_proj_index_mask_cache.npy').reshape(-1, num_retrieved)
test_lm_probs = np.load('saved_tensors/wikitext-103/test_lm_prob_cache.npy')

test_sample_ids = np.load('saved_tensors/wikitext-103/test_sample_id_cache.npy')

test_context_vecs = torch.from_numpy(test_context_vecs).float()
test_knns = torch.from_numpy(test_knns)
test_dists = torch.from_numpy(test_dists).float()
test_pkg_locality = torch.from_numpy(test_pkg_locality)
test_proj_locality = torch.from_numpy(test_proj_locality)
test_index_masks = torch.from_numpy(test_index_masks)
test_lm_probs = torch.from_numpy(test_lm_probs).float()

docs = []
for line in open('examples/language_model/wikitext103_seg/testtrain.txt'):
    docs.append(line.strip())

print(len(docs))
dictionary = torch.load('saved_tensors/wikitext-103/dictionary.pt')
targets = torch.load('saved_tensors/wikitext-103/test_original_tgts_cache.npy')
tgts = []
for t in targets:
    t = t.contiguous().view(-1)
    tgts.append(t[t != 1].cpu().numpy())

tgts = np.concatenate(tgts)


token_sample_map = torch.load('checkpoints/wikitext-103/testtrain_dstore_map.pt')
vals = np.memmap('checkpoints/wikitext-103/testtrain_dstore_vals.npy',
                                              dtype=np.int, mode='r',
                                              shape=(91912620, 1))

inv_token_sample_map = np.zeros(91912620, dtype='i')
for k, v in token_sample_map.items():
    inv_token_sample_map[v[0]:v[1]] = k

locality_indicator = test_proj_locality + 2 * test_pkg_locality

locality_feat = torch.nn.functional.one_hot(locality_indicator.long(), num_classes=4).permute(2, 0, 1)
modified_dists = locality_feat[0] * (1.2326 * test_dists) \
                 + locality_feat[1] * (1.2459 * test_dists + 1.0868) \
                 + locality_feat[2] * (1.2881 * test_dists + 1.2495) \
                 + locality_feat[3] * (1.2853 * test_dists + 1.4641)

probs = utils.log_softmax(modified_dists, dim=-1)

mod_knn_probs = torch.logsumexp(probs + test_index_masks, dim=-1)

orig_probs = utils.log_softmax(test_dists, dim=-1)
orig_knn_probs = torch.logsumexp(orig_probs + test_index_masks, dim=-1)

print(mod_knn_probs.shape)

knn_prob_diff = mod_knn_probs - orig_knn_probs
diffs, idxs = knn_prob_diff.topk(1000)
#
# print(diffs)
# print(idxs)

sample_ids = test_sample_ids[idxs]

print(sample_ids)

print(docs[sample_ids[989]])

print(orig_knn_probs[idxs][989])
print(diffs[989])
print("TARGET WORD", dictionary[tgts[idxs[989]].item()])
print('retrieved:')

for i, d, md in zip(test_knns[idxs, :5][989], orig_probs[idxs, :5][989],
                    probs[idxs, :5][989]):
    print(docs[inv_token_sample_map[i]])
    print(dictionary[int(vals[i])])
    print(d)
    print(md)

# print(locality_indicator[idxs, :5])
# print(test_dists[idxs, :5])
# print(modified_dists[idxs, :5])
