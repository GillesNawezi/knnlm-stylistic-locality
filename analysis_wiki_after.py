import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
import matplotlib.pyplot as plt
import seaborn as sns
from dask.diagnostics import ProgressBar

ProgressBar().register()

dists = np.load('saved_tensors/wikitext-103/test_modified_dist_cache.npy').flatten()
ranks = np.load('saved_tensors/wikitext-103/test_proj_rank_cache.npy')
pkg_locality = np.load('saved_tensors/wikitext-103/test_pkg_locality_cache.npy')
proj_locality = np.load('saved_tensors/wikitext-103/test_proj_locality_cache.npy')
correctness = np.load('saved_tensors/wikitext-103/test_proj_correctness_cache.npy')

locality = proj_locality + 2 * pkg_locality

dists = da.from_array(dists)
ranks = da.from_array(ranks)
locality = da.from_array(locality)
correctness = da.from_array(correctness)


arr_all = da.stack([dists, ranks, locality, correctness], axis=1)

ddf = dd.from_array(arr_all, columns=['dist', 'rank', 'locality', 'correctness'])
print(ddf)

# rank_filter_mask = ranks <= 64
#
# dists = dists[rank_filter_mask]
# ranks = ranks[rank_filter_mask]
# locality = locality[rank_filter_mask]

# dist - acc
bins = [-10000] + list(np.arange(-50, 0, 0.5)) + [0]
ddf['dist_range'] = ddf['dist'].map_partitions(pd.cut, bins)

dist_grouped = ddf.groupby(['locality', 'dist_range']).mean().reset_index().compute()

dist_grouped['dist_right'] = dist_grouped['dist_range'].apply(lambda x: x.right)

dist_grouped.to_csv('figures/wiki_dist_correctness_after.csv')

grouped = ddf.groupby(['locality', 'rank']).mean().reset_index().compute()
grouped.to_csv('figures/wiki_rank_after.csv')
