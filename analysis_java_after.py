import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
import matplotlib.pyplot as plt
import seaborn as sns
from dask.diagnostics import ProgressBar
ProgressBar().register()

dists = np.load('saved_tensors/java-huge-bpe-2000/test_modified_dist_cache.npy', mmap_mode='r').flatten()
ranks = np.load('saved_tensors/java-huge-bpe-2000/test_proj_rank_cache.npy', mmap_mode='r')
pkg_locality = np.load('saved_tensors/java-huge-bpe-2000/test_pkg_locality_cache.npy', mmap_mode='r')
proj_locality = np.load('saved_tensors/java-huge-bpe-2000/test_proj_locality_cache.npy', mmap_mode='r')
correctness = np.load('saved_tensors/java-huge-bpe-2000/test_proj_correctness_cache.npy', mmap_mode='r')

dists = da.from_array(dists)
ranks = da.from_array(ranks)
pkg_locality = da.from_array(pkg_locality)
proj_locality = da.from_array(proj_locality)
correctness = da.from_array(correctness)

project_local_only = (proj_locality == 1) & (pkg_locality == 0).astype('int8')
locality = project_local_only + 2*pkg_locality
arr_all = da.stack([dists, ranks, locality, correctness], axis=1)

ddf = dd.from_array(arr_all, columns=['dist', 'rank', 'locality', 'correctness'])
print(ddf)
print('df build complete')

# rank_filter_mask = ranks <= 64
#
# dists = dists[rank_filter_mask]
# ranks = ranks[rank_filter_mask]
# locality = locality[rank_filter_mask]

# bins = [-10000] + list(range(-500, 0, 10)) + [0]
# ddf['dist_range'] = ddf['dist'].map_partitions(pd.cut, bins)
#
# dist_grouped = ddf.groupby(['locality', 'dist_range']).mean().reset_index().compute()
#
# dist_grouped['dist_right'] = dist_grouped['dist_range'].apply(lambda x: x.right)
#
# dist_grouped.to_csv('figures/java_dist_correctness_after.csv')

grouped = ddf.groupby(['locality', 'rank']).mean().reset_index().compute()

grouped.to_csv('figures/java_rank_after.csv')

