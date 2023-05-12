# %% import libraries
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
#from umap import UMAP

import pandas as pd

# %% Loading features

filez = np.load('features.npz')
features = filez['features']
dataset = filez['dataset']
tache = filez['training_task']

# %% reduce dimension with PCA=>t-SNE
reduction = "PCA+T-SNE"

pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=30))])
features_pca = pipeline.fit_transform(features)
print(f'Variance explained: {pipeline[1].explained_variance_ratio_.sum()}')

tsne = TSNE(n_components = 2)
features_tsne = tsne.fit_transform(features_pca)

df_features = pd.DataFrame(features_tsne, columns=['x', 'y'])
df = pd.read_csv('attributes.csv', index_col=0)
df = pd.concat([df, df_features], axis=1)

df.to_csv(f"../data/csv/{dataset}_{tache}_{reduction}.csv", index=False)

def reduce_dimensions(method, n_components, features, df_metadata):
    if method == "PCA":
      pipe = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=n_components))])
      features_transformed = pipe.fit_transform(features)
      
    if method == "TSNE":
      pipe = Pipeline([('scaling', StandardScaler()), ('t-sne', TSNE(n_components=n_components))])
      features_transformed = pipe.fit_transform(features)
      
    if method == "UMAP":
      umap_2d = umap.UMAP(n_components=2, random_state=0)
      umap_2d.fit(features)
      features_transformed = umap_2d.transform(features)
    
    df_features = pd.DataFrame(features_transformed, columns=['x','y'])
    df = pd.concat([df_metadata, df_features], axis=1)
    return df


# %% reduce dimension with PCA
reduction = "PCA"
n_components = 2
df_metadata = pd.read_csv('attributes.csv', index_col=0)
df_pca = reduce_dimensions('PCA', n_components, features, df_metadata)
df_pca.to_csv(f"../data/csv/{dataset}_{tache}_{reduction}.csv", index=False)

# %% reduce dimension with t-SNE
reduction = "t-SNE"
n_components = 2
df_metadata = pd.read_csv('attributes.csv', index_col=0)
df_tsne = reduce_dimensions('TSNE', n_components, features, df_metadata)
df_tsne.to_csv(f"../data/csv/{dataset}_{tache}_{reduction}.csv", index=False)

# %% reduce dimension with UMAP
reduction = "UMAP"
n_components = 2
df_metadata = pd.read_csv('attributes.csv', index_col=0)
df_umap = reduce_dimensions('UMAP', n_components, features, df_metadata)
df_umap.to_csv(f"../data/csv/{dataset}_{tache}_{reduction}.csv", index=False)