import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dists = np.load('/node09_data/frank/test_dist_cache.npy')
ranks = np.load('/node09_data/frank/test_rank_cache.npy')
locality = np.load('/node09_data/frank/test_locality_cache.npy')

rank_filter_mask = ranks <= 100

dists = dists[rank_filter_mask]
ranks = ranks[rank_filter_mask]
locality = locality[rank_filter_mask]

dist_filter_mask = dists > -50
dists = dists[dist_filter_mask]
ranks = ranks[dist_filter_mask]
locality = locality[dist_filter_mask]

print('all', len(dists))
local_mask = locality == 1
non_local_mask = locality == 0
local_dists = dists[local_mask]
local_ranks = ranks[local_mask]
non_local_dists = dists[non_local_mask]
non_local_ranks = ranks[non_local_mask]
print('local', len(local_dists))
print('non-local', len(non_local_dists))


n_bins = 150

fig, ax = plt.subplots(figsize=(8, 4))

n, bins, patches = ax.hist(non_local_dists, n_bins, density=True, histtype='step',
                           cumulative=True, label='Non Local')

n, bins, patches = ax.hist(local_dists, n_bins, density=True, histtype='step',
                           cumulative=True, label='Local')
ax.grid(True)
ax.legend(loc='center left')
ax.set_title('Cumulative step histograms')
ax.set_xlabel('Negative distance')
ax.set_ylabel('Likelihood of occurrence')

plt.savefig('package_local_cdf_100_cutdist50.pdf')
