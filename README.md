# The pipeline
1. Train the model with ```train_resnet50.py```. This will save the best epoch in ```checkpoint.torch```.
2. Export features with ```predict_resnet50.py```. This will save an ```attributes.csv``` containing infos about all the tasks of the current dataset used and if the image was used in train or test for each features and a matrix ```features.npy```.
3. Reduce feature dimension with ```dim_reduction.py```. The result will be csv files for every reduction dimension chosen. They will have this form ```{dataset}_{task}_{dim_reduction}.csv``` . Each of them will be consisting in ```attributes.csv``` completed with the two selected features from ```features.npy``` after dimension reduction applied on it. 
4. Plot the features with selected colors with ```feature_plots.py```.

# Utilitaries
The file ```train_util.py``` contains utilitaries functions mostly used to manipulate the neural network.

# Constructing the dataset
The file ```construct_splits``` permits to construct train and val folder for each task containing subfolder for each class. This enables the use of ImageFolder from PyTorch. The function does not use extraspace as it makes use of hard links.

# Wikiart 

- Download Wikiart [here](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)

- Put your Wikiart images in 
```bash
static/images/wikiart/data
```