import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
import matplotlib.pyplot as plt
import seaborn as sns
from dask.diagnostics import ProgressBar

ProgressBar().register()

dists = np.load('saved_tensors/wikitext-103/test_proj_dist_cache.npy')
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

ddf = ddf[ddf['dist'] >= -15]

ddf = ddf.sort_values(['dist']).reset_index(drop=True)


ddf['overall_rank'] = ddf.groupby('locality').cumcount()


# dist - acc
bins = list(np.arange(0, 108794826, 50000))
ddf['rank_range'] = ddf['overall_rank'].map_partitions(pd.cut, bins)

dist_grouped = ddf.groupby(['locality', 'rank_range']).mean().reset_index().compute()

# dist_grouped['dist_right'] = dist_grouped['dist_range'].apply(lambda x: x.right)

dist_grouped.to_csv('figures/wiki_dist_correctness.csv')


fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(x='dist', y='correctness', hue='locality', data=dist_grouped)

plt.savefig('figures/wiki_avg_correctness_by_dist_1024.pdf')

exit()

# rank - acc
grouped = ddf.groupby(['locality', 'rank']).mean().reset_index().compute()
grouped.to_csv('figures/wiki_rank.csv')

fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(x='rank', y='correctness', hue='locality', data=grouped, s=5)

plt.savefig('figures/wiki_avg_correctness_by_rank_1024.pdf')

# rank - dist

fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(x='rank', y='dist', hue='locality', data=grouped, s=5)

plt.savefig('figures/wiki_avg_dist_by_rank_1024.pdf')
