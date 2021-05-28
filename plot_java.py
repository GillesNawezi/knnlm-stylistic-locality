import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette('colorblind', n_colors=3)
# dist - acc
dist_grouped = pd.read_csv('figures/java_dist_correctness.csv')
dist_grouped = dist_grouped[dist_grouped['dist_right'] >= -400]


conditions = [
    (dist_grouped['locality'] == 0),
    (dist_grouped['locality'] == 1),
    (dist_grouped['locality'] == 2)]
choices = ['no locality',
           'same project, different subdir',
           'same subdir']

dist_grouped['Locality'] = np.select(conditions, choices)

dist_grouped['Accuracy'] = dist_grouped['correctness']
dist_grouped['Neg. Distance'] = dist_grouped['dist_right']

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
sns.scatterplot(x='Neg. Distance', y='Accuracy', hue='Locality', data=dist_grouped, s=11,
                palette=color, ax=ax[0], legend=False)

grouped = pd.read_csv('figures/java_rank.csv')
grouped = grouped.loc[grouped['rank'] <= 200]
conditions = [
    (grouped['locality'] == 0),
    (grouped['locality'] == 1),
    (grouped['locality'] == 2)]

grouped['Locality'] = np.select(conditions, choices)
grouped['Rank'] = grouped['rank']

grouped['Accuracy'] = grouped['correctness']
grouped['Neg. Distance'] = grouped['dist']

# rank - acc
sns.scatterplot(x='Rank', y='Accuracy', hue='Locality', data=grouped, s=8,
                palette=color, ax=ax[1], legend=False)

# rank - dist
sns.scatterplot(x='Rank', y='Neg. Distance', hue='Locality', data=grouped, s=8,
                palette=color, ax=ax[2])

fig.tight_layout()
plt.savefig('figures/java.pdf')
