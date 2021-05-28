import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette('colorblind', n_colors=4)
# dist - acc
dist_grouped = pd.read_csv('figures/wiki_dist_correctness_after.csv')

conditions = [
    (dist_grouped['locality'] == 0),
    (dist_grouped['locality'] == 1),
    (dist_grouped['locality'] == 2),
    (dist_grouped['locality'] == 3)]
choices = ['no locality',
           'same category, different section',
           'same section, different category',
           'same section, same category']

dist_grouped['Locality'] = np.select(conditions, choices)

dist_grouped['Accuracy'] = dist_grouped['correctness']
dist_grouped['Neg. Distance'] = dist_grouped['dist_right']

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
# sns.scatterplot(x='Neg. Distance', y='Accuracy', hue='Locality', data=dist_grouped, s=8,
#                 palette=color, ax=ax[0], legend=False)

grouped = pd.read_csv('figures/wiki_rank_after.csv')
grouped = grouped.loc[grouped['rank'] <= 200]
conditions = [
    (grouped['locality'] == 0),
    (grouped['locality'] == 1),
    (grouped['locality'] == 2),
    (grouped['locality'] == 3)]

grouped['Locality'] = np.select(conditions, choices)
grouped['Rank'] = grouped['rank']

grouped['Accuracy'] = grouped['correctness']
grouped['Neg. Distance (Modified)'] = grouped['dist']

# rank - acc
# sns.scatterplot(x='Rank', y='Accuracy', hue='Locality', data=grouped, s=8,
#                 palette=color, ax=ax[1], legend=False)

# rank - dist
sns.scatterplot(x='Rank', y='Neg. Distance (Modified)', hue='Locality', data=grouped, s=8,
                palette=color)

fig.tight_layout()
plt.savefig('figures/wiki_after.pdf')
