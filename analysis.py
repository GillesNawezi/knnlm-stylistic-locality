import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# dists = np.load('/node09_data/frank/test_dist_cache.npy')
# ranks = np.load('/node09_data/frank/test_rank_cache.npy')
# locality = np.load('/node09_data/frank/test_locality_cache.npy')
#
# # use all locality=1 data for more samples..
# print('all', len(dists))
# local_dists = dists[locality == 1]
# local_ranks = ranks[locality == 1]
# non_local_dists = dists[locality == 0]
# non_local_ranks = ranks[locality == 0]
# print('local', len(local_dists))
# print('non-local', len(non_local_dists))
#
# # Generate the permutation index array.
# permutation = np.random.permutation(non_local_dists.shape[0])
#
# np.save('analysis_data/dist_smpl.npy', np.concatenate((non_local_dists[permutation][:10000000], local_dists)))
# np.save('analysis_data/rank_smpl.npy', np.concatenate((non_local_ranks[permutation][:10000000], local_ranks)))
# np.save('analysis_data/locality_smpl.npy', np.concatenate((np.zeros(10000000), np.ones(len(local_dists)))))
#
# exit()

# dists = np.load('analysis_data/dist_smpl.npy')
# ranks = np.load('analysis_data/rank_smpl.npy')
# locality = np.load('analysis_data/locality_smpl.npy')

# load smaller K's data
dists = np.load('/node09_data/frank/test_proj_dist_cache.npy')
ranks = np.load('/node09_data/frank/test_proj_rank_cache.npy')
locality = np.load('/node09_data/frank/test_pkg_locality_cache.npy')
correctness = np.load('/node09_data/frank/test_proj_correctness_cache.npy')

# rank_filter_mask = ranks <= 64
#
# dists = dists[rank_filter_mask]
# ranks = ranks[rank_filter_mask]
# locality = locality[rank_filter_mask]


df = pd.DataFrame({'dist': dists, 'rank': ranks, 'locality': locality})
# print(df.groupby(['locality', 'rank']).count().reset_index())

grouped = df.groupby(['locality', 'rank']).mean().reset_index()


proj_dists = np.load('/node09_data/frank/test_proj_dist_cache.npy')
proj_ranks = np.load('/node09_data/frank/test_proj_rank_cache.npy')
proj_locality = np.load('/node09_data/frank/test_proj_locality_cache.npy')

df2 = pd.DataFrame({'dist': proj_dists, 'rank': proj_ranks, 'locality': proj_locality})
grouped2 = df2.groupby(['locality', 'rank']).mean().reset_index()

grouped.loc[grouped['locality'] == 1, 'locality'] = 2
grouped1_filtered = grouped[grouped['locality'] == 2]

grouped = grouped2.append(grouped1_filtered)

print(grouped)

fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(x='rank', y='dist', hue='locality', data=grouped, s=5)

plt.savefig('package_local_avg_dist_by_rank_64.pdf')

print(grouped)
# cumulative histogram

# n_bins = 150
#
# fig, ax = plt.subplots(figsize=(8, 4))
#
# n, bins, patches = ax.hist(non_local_dists, n_bins, density=True, histtype='step',
#                            cumulative=True, label='Non Local')
#
# n, bins, patches = ax.hist(local_dists, n_bins, density=True, histtype='step',
#                            cumulative=True, label='Local')
# ax.grid(True)
# ax.legend(loc='center left')
# ax.set_title('Cumulative step histograms')
# ax.set_xlabel('Negative distance')
# ax.set_ylabel('Likelihood of occurrence')
#
# plt.savefig('package_local_cdf.pdf')

# print('Local = 0')
# print('N:', len(non_local_dists))
# r1 = np.mean(non_local_dists)
# print("Mean:", r1)
# r2 = np.median(non_local_dists)
# print("Median:", r2)
# print(np.quantile(non_local_dists, 0.9))
# print(np.quantile(non_local_dists, 0.1))
#
# print('Local = 1')
# print('N:', len(local_dists))
# r1 = np.mean(local_dists)
# print("Mean:", r1)
# r2 = np.median(local_dists)
# print("Median:", r2)
# print(np.quantile(local_dists, 0.9))
# print(np.quantile(local_dists, 0.1))
