# %% import libraries
import os
from pathlib import Path
import pandas as pd

# %% construct splits
dataset = 'wikiart/'

dataset_dir = Path('/home/art_ter/pytorch/' + dataset)
dataset_dir = Path('../static/images/' + dataset)


task_folder = 'tasks'

print(os.getcwd())


print(f'Creating folder : {dataset_dir / task_folder}')
os.mkdir(dataset_dir / task_folder)
 
metadata = [fichier for fichier in os.listdir(
        dataset_dir / 'metadata/') if fichier.endswith('.csv')]

tasks = list(set([fichier.split('_')[0] for fichier in metadata ]))

for task in tasks:
    curr_path = dataset_dir / task_folder / task
    # creating task folder
    print(f'Creating folder : {curr_path}')
    os.mkdir(curr_path)
    df = pd.read_csv(dataset_dir / 'metadata' / f'{task}_class.txt',
                     sep=' ', names=['id', 'label'])\
           .set_index('id')

    mapping = df.to_dict()['label']
    for split in ('train', 'val'):
        # creating folders for current split
        print(f'Creating folder : {curr_path / split}')
        os.mkdir(curr_path / split) 
        # creating classes for the current task
        for label in df.label:
            print(f'Creating folder : {curr_path / split / label}')
            os.mkdir(curr_path / split / label)

        df_split = pd.read_csv(dataset_dir / 'metadata' / f'{task}_{split}.csv',
                               header= None,
                               names=['img', 'label'], encoding='utf-8')
        for _, row in df_split.iterrows():
            img_from = dataset_dir / 'data' / row.img
            label = mapping[row.label]
            img_to = curr_path / split / label / img_from.name
            print(f'{img_from} => {img_to}')
            os.link(img_from, img_to)
