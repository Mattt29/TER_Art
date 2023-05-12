# %% import libraries
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# %% load data
df = pd.read_csv('data.csv')
df.rename({'feature_1': 'Feature 1', 'feature_2': 'Feature 2'}, axis=1, inplace=True)
# %% plot
tasks = ('style', 'genre', 'artist')
subsets = ('train', 'val')
fig, axs = plt.subplots(nrows=len(tasks),
                        ncols=1,
                        figsize=(8, 12),
                        constrained_layout=True)

for ax in axs:
    ax.remove()

gridspec = axs[0].get_subplotspec().get_gridspec()
subfigs = [fig.add_subfigure(gs) for gs in gridspec] 

for row, subfig in enumerate(subfigs):
    dft = df[~df[f'{tasks[row]}_name'].isna()]
    hue_order = dft[f'{tasks[row]}_name'].unique()

    subfig.suptitle(f'Coloring: {tasks[row]}')

    axs = subfig.subplots(nrows=1, ncols=2)

    for col, ax in enumerate(axs):
        dfp = dft[dft.subset == subsets[col]]
        ax.set_title(f'Subset: {subsets[col]}')
        ax.axis('off')
        sns.scatterplot(data=dfp.sample(n=700),  # no need to plot everything...
                        x='Feature 1',
                        y='Feature 2',
                        hue=f'{tasks[row]}_name',
                        hue_order=hue_order,
                        style=f'{tasks[row]}_name',
                        style_order=hue_order,
                        ax=ax,
                        legend=False)
plt.suptitle('Feature space', size=24)
plt.savefig('feature_space.jpeg', dpi=300)
plt.show()

# %%
