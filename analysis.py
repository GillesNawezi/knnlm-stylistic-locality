import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dists = np.load('/node09_data/frank/test_proj_dist_cache.npy')
ranks = np.load('/node09_data/frank/test_proj_rank_cache.npy')
pkg_locality = np.load('/node09_data/frank/test_pkg_locality_cache.npy')
proj_locality = np.load('/node09_data/frank/test_proj_locality_cache.npy')

print(dists.shape)
# rank_filter_mask = ranks <= 64
#
# dists = dists[rank_filter_mask]
# ranks = ranks[rank_filter_mask]
# locality = locality[rank_filter_mask]


df = pd.DataFrame({'dist': dists, 'rank': ranks,
                   'pkg_locality': pkg_locality, 'proj_locality': proj_locality})

# create a list of our conditions
conditions = [
    (df['proj_locality'] == 0),
    (df['proj_locality'] == 1) & (df['pkg_locality'] == 0),
    (df['pkg_locality'] == 1),
    ]

# create a list of the values we want to assign for each condition
values = [0, 1, 2]

# create a new column and use np.select to assign values to it using our lists as arguments
df['locality'] = np.select(conditions, values)

grouped = df.groupby(['locality', 'rank']).mean().reset_index()

print(grouped)

fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(x='rank', y='dist', hue='locality', data=grouped, s=5)

plt.savefig('package_local_avg_dist_by_rank_1024.pdf')
