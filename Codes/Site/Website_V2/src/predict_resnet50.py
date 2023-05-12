# %% import libraries
if __name__ == '__main__':
    import os
    from pathlib import Path

    from torch.utils.data import DataLoader

    from torchvision import datasets

    import torch 

    from timm.data.transforms_factory import create_transform
    from timm import create_model

    import numpy as np

    import pandas as pd

    from PIL import ImageFile

    from train_util import load_checkpoint, predict

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    # %% configuration
    #data_dir = Path('/home/data/wikiart/')
    #data_dir = Path('/home/art_ter/pytorch/wiki_art')
    #dataset = 'style'  # contains all painting

    dataset_name = 'wikiart'
    #training_task = 'nationality'
    #training_task = 'genre'
    training_task = 'artist'
    training_task = 'style'
        
    dataset_dir = Path('/home/art_ter/pytorch/' + dataset_name)
    dataset_dir = Path('../static/images/' + dataset_name)


    input_size = 224

    transform = create_transform(input_size, is_training=False)

    dataset = datasets.ImageFolder(
        dataset_dir / 'data',
        transform=transform
    )

    batch_size = 128
    num_workers = 12

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    n_classes = len(os.listdir(dataset_dir / 'tasks' / training_task / 'train'))

    device_id = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    #device = 'cpu'

    model = create_model('resnet50', num_classes=n_classes)

    #model.load_state_dict(torch.load(PATH, map_location=device))

    load_checkpoint(model)
    
    model.to(device)
    #model.to(f'cuda:{model.device_ids[0]}')

    # %% predicting logits
    # logits = predict(loader, model, device_id)

    # %% predicting features (WHAT WE ACTUALLY WANT)
    model.reset_classifier(0)  # because Timm created the model
    
    features = predict(loader, model, device, device_id)

    np.savez('features.npz', features = features, dataset = np.array(dataset_name), training_task = np.array(training_task))
    
    # %% constructing attributes
    imgs = pd.DataFrame([p[0] for p in dataset.samples], columns=['name'])
    imgs.name = imgs.name.apply(lambda p: f'{Path(p).parent.name}/{Path(p).name}')

    metadata = [fichier for fichier in os.listdir(
        dataset_dir / 'metadata/') if fichier.endswith('.csv')]

    tasks = list(set([fichier.split('_')[0] for fichier in metadata ]))

    for t in tasks:
        df_class = pd.read_csv(dataset_dir / 'metadata' / f'{t}_class.txt',
                            header=None,
                            names=['idx', f'{t}_name'],
                            sep=' ').set_index('idx')

        df_train = pd.read_csv(dataset_dir / 'metadata' / f'{t}_train.csv',
                            header=None,
                            names=['file_name', 'idx'])
        if t == training_task:
            df_train['subset'] = 'train'
        df_val = pd.read_csv(dataset_dir / 'metadata' / f'{t}_val.csv',
                            header=None,
                            names=['file_name', 'idx'])
        if t == training_task:
            df_val['subset'] = 'val'
        df_all = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
        df_t = df_all.join(df_class, how='left', on='idx')\
                    .drop('idx', axis=1)\
                    .set_index('file_name')
        imgs = imgs.join(df_t, how='left', on='name')

    imgs.to_csv('attributes.csv', index=True)
    print('fin')