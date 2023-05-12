# %% import libraries
import os
from pathlib import Path
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from torchvision import datasets

from timm.data.transforms_factory import create_transform
from timm import create_model

from PIL import ImageFile

from train_util import train, test, learning_curves, save_checkpoint

ImageFile.LOAD_TRUNCATED_IMAGES = True

# %% configuration
dataset = 'wikiart'
task = 'style'
#task = 'nationality'
#task = 'genre'
#task = 'artist'

dataset_dir = Path('/home/art_ter/pytorch/' + dataset + '/')
dataset_dir = Path('../static/images/' + dataset + '/')


metadata = [fichier for fichier in os.listdir(
        dataset_dir / 'metadata/') if fichier.endswith('.csv')]

tasks = list(set([fichier.split('_')[0] for fichier in metadata ]))


task_id = tasks.index(task) # the task we want to train our model on

input_size = 224
transform_train = create_transform(input_size, is_training=True)
transform_test = create_transform(input_size, is_training=False)

batch_size = 64

train_set = datasets.ImageFolder(
    dataset_dir / 'tasks' / tasks[task_id] / 'train',
    transform=transform_train
)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=16
)

test_set = datasets.ImageFolder(
    dataset_dir / 'tasks' / tasks[task_id] / 'val',
    transform=transform_train
)

test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=16
)

n_classes = len(os.listdir(dataset_dir / 'tasks' / tasks[task_id] / 'train'))

device_id = 0

model = create_model('resnet50',
                     pretrained=True,
                     num_classes=n_classes,
                     drop_rate=0.2)


#torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(device)


if(torch.cuda.device_count() > 1 and device=='cuda'):
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)
  model.to(f'cuda:{model.device_ids[0]}')
  
elif(device == 'cuda'):
  model.cuda(device_id)


#model = nn.DataParallel(model, [0,1])
# %% model summary
# print(summary(model,input_size=(32, 3, 224, 224)))

# %% optimization configuration
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005)

scheduler = MultiStepLR(optimizer, milestones=[15, 20], gamma=0.1)

# %% optimization
y_loss = {
    'train': [],
    'test': []
}

y_acc = {
    'train': [],
    'test': []
}

best_acc = 0

num_epochs = 25
for epoch in range(num_epochs):  # loop over the dataset multiple times
    print(f'------------------ Training Epoch {epoch+1} ------------------')
    # train loop
    loss, acc = train(train_loader, model, loss_fn, optimizer, device, device_id)
    y_loss['train'].append(loss)
    y_acc['train'].append(acc)

    # test loop
    loss, acc = test(test_loader, model, loss_fn, device, device_id)
    y_loss['test'].append(loss)
    y_acc['test'].append(acc)

    learning_curves(y_loss, y_acc)

    if acc > best_acc:
        best_acc = acc
        save_checkpoint(model)

    # update the learning rate
    scheduler.step()
