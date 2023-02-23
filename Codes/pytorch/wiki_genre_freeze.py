import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import ToTensor, Lambda, Compose
import os,time
from torch.utils.data import random_split, TensorDataset 
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchinfo import summary
import matplotlib.pyplot as plt


def main():
    data_dir = 'D:/art/wikiartorigine_/wikiart' 
    #data_dir = '/home/art_ter/pytorch/wikiart'

    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.CenterCrop((224, 224)),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    wikiart_dataset = datasets.ImageFolder(os.path.join(data_dir), transform=transform)
    
    batch_size = 32
    train_size = int(0.7 * len(wikiart_dataset))
    test_size = len(wikiart_dataset) - train_size
    train_data, test_data = random_split(wikiart_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=2) #shuffle?
    test_dataloader = DataLoader(test_data, batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    #### Avec timm : ###
    """
    import timm
    model = timm.create_model('resnet18', pretrained=True, num_classes=27, drop_rate = 0.5)
    """
    #### Avec torchivision.models : ####
    model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    ## essai avec juste la dernière couche changée pour retourner 27 classes
    """
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 27)          

    model=model.to(device)
    
    print(summary(model,input_size=(32,3,224,224)))

    """
    ## essai avec 2 dernières couches ajoutées pour finir sur 27 classes
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=2048, out_features=1024, bias=True),
        torch.nn.Linear(in_features=1024, out_features=27, bias=True)
        ).to(device)

    #print(model)
    #print(summary(model,input_size=(32,3,224,224)))
    """
    print(summary(model, 
        input_size=(32, 3, 224, 224), 
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    ))
    """
    model=model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)               # essai de plusieurs optimizers..
    
    #scheduler = MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)

    ### plot des loss et accuracy au fil des epochs
    y_loss = {} 
    y_loss['train'] = []
    y_loss['test'] = []
    y_acc = {}
    y_acc['train'] = []
    y_acc['test'] = []

    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="accuracy")


    def draw_curve(current_epoch):
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
        ax0.plot(x_epoch, y_loss['test'], 'ro-', label='test')
        ax1.plot(x_epoch, y_acc['train'], 'bo-', label='train')
        ax1.plot(x_epoch, y_acc['test'], 'ro-', label='test')
        if current_epoch == 0:
            ax0.legend()
            ax1.legend()
        fig.savefig(os.path.join('/home/art_ter/pytorch/lossGraphs/', 'train.jpg'))

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        running_loss = 0.0
        running_corrects = 0.0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            print(batch)
            # Compute prediction error
            pred = model(X)
            #print(np.shape(pred),np.shape(y))
            _, preds = torch.max(pred.data, 1)
            loss = loss_fn(pred, y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size
            running_corrects += float(torch.sum(preds == y.data))

            if batch % 500 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        epoch_loss = running_loss / size
        epoch_acc = running_corrects / size

        y_loss['train'].append(epoch_loss)
        y_acc['train'].append(epoch_acc)


    def test(dataloader, model):
        size = len(dataloader.dataset)
        model.eval()
        test_loss, correct = 0, 0
        i=0
        with torch.no_grad():
            for X, y in dataloader:
                i+=32
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                if i % 16000 == 0: 
                    print(f'[{i}/{size}] correct : {correct}, total : {i}, ratio : {correct/i}')

        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss*batch_size:>8f} \n")


        y_loss['test'].append(test_loss*batch_size)
        y_acc['test'].append(correct)
        draw_curve(epoch)
        return 100*correct


    def save_checkpoint(state, is_best, filename='/home/art_ter/pytorch/model/wikigenrefreeze/checkpoint.pth.tar'):
        """Save checkpoint if a new best is achieved"""
        if is_best:
            print ("=> Saving a new best")
            torch.save(state, filename)  # save checkpoint
        else:
            print ("=> Validation Accuracy did not improve")

    best_accuracy = 0 
    num_epochs = 15
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("------------------ Training Epoch {} ------------------".format(epoch+1))

        train(train_dataloader, model, loss_fn, optimizer)
        
        acc=test(test_dataloader, model)

        is_best = bool(acc > best_accuracy)
       
        best_accuracy = max(acc, best_accuracy)
        # Save checkpoint if is a new best
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_accuracy': best_accuracy
        }, is_best,filename=f'/home/art_ter/pytorch/model/wikigenrefreeze/bestcheckpoint.pth.tar')
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_accuracy': acc
        }, True,filename=f"/home/art_ter/pytorch/model/wikigenrefreeze/checkpoint-{epoch+1}.pth.tar")
    print('Finished Training')


    torch.save(model.state_dict(), '/home/art_ter/pytorch/output/wiki_genre_freeze_trained.torch')
    print("Saved PyTorch Model State to wiki_genre_trained.torch")

    
    
    model_trained = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model_trained.fc.in_features
    model_trained.fc = nn.Linear(num_ftrs, 27)
    model_trained.load_state_dict(torch.load('/home/art_ter/pytorch/output/wiki_genre_freeze_trained.torch'))
    model_trained.fc = nn.Identity()

    import cv2 

    def predict(model, loader, criterion=nn.CrossEntropyLoss(), feature_extract=False, max_size=0, resize=128):
        with torch.no_grad():
            if not feature_extract:
                model.eval()

            y_preds = []
            y_labels = []
            inputs_ = []

            running_loss = 0.0
            size = 0.0
            for idx, data in enumerate(loader):
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                # wrap them in Variable
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                y_preds.extend(outputs.data.tolist())
                y_labels.extend(labels.data.tolist())
                if size <= max_size and feature_extract:
                    images = [
                        cv2.resize(
                            (
                                (i*0.224+0.456)*255).astype('uint8').transpose((1, 2, 0)), dsize=(resize, resize)
                        ) for i in inputs.data.cpu().numpy()
                    ]
                    inputs_.extend(images)
                    size = len(inputs_)

            predictions, labels, inputs = np.asarray(y_preds), np.asarray(y_labels), np.asarray(inputs_)

        if not feature_extract:
            return predictions, labels, running_loss/len(loader)

        return predictions, labels, inputs

    features, labels, images = predict(
        model, 
        train_dataloader, 
        feature_extract=True, 
        max_size=len(train_data)
    )

    print(features.shape, labels.shape, images.shape)
    torch.save(features, '/home/art_ter/pytorch/output/wiki_genre_freeze_features.pt')
    
    torch.save(labels, '/home/art_ter/pytorch/output/wiki_genre_freeze_labels.pt')

    print("Saved PyTorch feature space to wiki_genre__freeze_features.torch")

if __name__ == '__main__':
    main()

    

