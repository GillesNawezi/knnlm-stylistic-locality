import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load smaller K's data
dists = np.load('/node09_data/frank/test_proj_dist_cache.npy')
ranks = np.load('/node09_data/frank/test_proj_rank_cache.npy')
pkg_locality = np.load('/node09_data/frank/test_pkg_locality_cache.npy')
proj_locality = np.load('/node09_data/frank/test_proj_locality_cache.npy')
correctness = np.load('/node09_data/frank/test_proj_correctness_cache.npy')

df = pd.DataFrame({'dist': dists, 'rank': ranks, 'correctness': correctness,
                   'pkg_locality': pkg_locality, 'proj_locality': proj_locality})
df = df[df['dist'] >= -150]
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

# #bin 100
df['dist_range'] = pd.cut(df['dist'], 100)

dist_grouped = df.groupby(['locality', 'dist_range']).mean().reset_index()

dist_grouped['dist_right'] = dist_grouped['dist_range'].apply(lambda x: x.right)

dist_grouped.to_csv('dist_correctness.csv')

fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(x='dist_right', y='correctness', hue='locality', data=dist_grouped, s=5)

plt.savefig('avg_correctness_by_dist_64.pdf')


grouped = df.groupby(['locality', 'rank']).mean().reset_index()

print(grouped)

fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(x='rank', y='correctness', hue='locality', data=grouped, s=5)

plt.savefig('avg_correctness_by_rank_64.pdf')


