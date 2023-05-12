import os
import warnings

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import Tuple

import matplotlib.pyplot as plt

# from tqdm import tqdm

from PIL.Image import DecompressionBombWarning

warnings.simplefilter("ignore", DecompressionBombWarning)


def train(dataloader:DataLoader,
          model:Module,
          loss_fn:_Loss,
          optimizer:Optimizer,
          device,
          device_id:int) -> Tuple[float, float]:
    size = len(dataloader.dataset)
    running_loss = 0.0
    running_corrects = 0.0

    # putting the model on train mode for dropout, batchnorm, etc.
    model.train()

    # looping for all batches
    for batch, (X, y) in enumerate(dataloader): #tqdm(enumerate(dataloader)):
        if device == 'cuda':
            X, y = X.cuda(device_id), y.cuda(device_id)
        #X, y = X.to(f'cuda:{model.device_ids[0]}'), y.to(f'cuda:{model.device_ids[0]}')
        # compute loss
        output = model(X)
        loss = loss_fn(output, y)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # monitoring loss
        running_loss += loss.item() * X.size(0)

        # monitoring accuracy
        _, preds = torch.max(output, 1)
        running_corrects += float(torch.sum(preds == y))

        # logging every 100 batches
        if batch % 100 == 99:
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"loss: {loss:>4f}  [{current:>5d}/{size:>5d}]")

    # normalizing loss and accuracy per element
    epoch_loss = running_loss / size
    epoch_acc = running_corrects / size
    return epoch_loss, epoch_acc


def test(dataloader:DataLoader,
         model:Module,
         loss_fn:_Loss, device,
         device_id:int) -> Tuple[float, float]:
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    i=0
    # we avoid computing the gradient graph
    with torch.no_grad():
        for j, (X, y) in enumerate(dataloader):
            i+=X.size(0)
            if device == 'cuda':
                X, y = X.cuda(device_id), y.cuda(device_id)
            output = model(X)
            test_loss += loss_fn(output, y).item() * X.size(0)
            correct += (output.argmax(1) == y).sum().item()

            # if j % 50 == 49:
            #    print(f'[{i}/{size}] correct : {correct}, total : {i}, ratio : {correct/i}')

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>5f}\n")

    return test_loss, correct


def predict(dataloader:DataLoader,
            model:Module, device,
            device_id:int) -> Tensor:
    size = len(dataloader.dataset)
    model.eval()
    
    #print(torch.cuda.is_available())
    #print(torch.cuda.current_device())
    #print(torch.cuda.get_device_name(0))
    
    
    
    i=0
    results = []
    # we avoid computing the gradient graph
    with torch.no_grad():
        for j, (X, y) in enumerate(dataloader):
            i+=X.size(0)
            if device == 'cuda':
                X, y = X.cuda(device_id), y.cuda(device_id)
            output = model(X)

            results.append(output.cpu())
            if j % 50 == 49:
                print(f'[{i}/{size}]')

    results = torch.concat(results, axis=0)

    return results

def save_checkpoint(model:Module, filename:str='checkpoint.torch') -> None:
    torch.save(model.state_dict(), filename)  # save checkpoint


def load_checkpoint(model:Module, filename:str='checkpoint.torch') -> None:
    model.load_state_dict(torch.load(filename))


def learning_curves(y_loss:dict, y_acc:dict) -> None:
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="accuracy")

    x = [i+1 for i in range(len(y_loss['train']))]

    ax0.plot(x, y_loss['train'], 'bo-', label='train')
    ax0.plot(x, y_loss['test'], 'ro-', label='test')
    ax1.plot(x, y_acc['train'], 'bo-', label='train')
    ax1.plot(x, y_acc['test'], 'ro-', label='test')

    ax0.legend()
    ax1.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(os.getenv('HOME'), 'train.jpg'))
