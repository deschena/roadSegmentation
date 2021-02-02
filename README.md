# ML-Project-2: Road Segmentation 
*This project was realized during the Machine Learning class at EPFL (CS-433). The goal was to perform semantic segmentation on aerial images to classify road and background pixels.*

*In case this repo was helpful to you, leaving a star is much appreciated. Thanks.*

#### Final results (computed on unseed data, on submission server)
* F1 score: 0.907
* Accuracy: 0.950


#### Requirements

- PyTorch 1.7.0 (with cuda 11.0)


## Datasets

This folder contains all the data used to train the model and the functions used to augment the data and create the augmented dataset. Namely : 
* datasets/augmented_dataset : the folder of all images used for training once augmented
* datasets/massachussetts_dataset : the other dataset found on Kaggle
* training : the original training dataset provided
* AugmDataset.py : the class used to augment the data
* ToLabelDataset.py : a loader for images to label
* create_dataset.py : creates the datasets used for the experiments. We merged "train_clean" and the massachussets_dataset folders to obtain "train_base_n_massachussetts". Dataset  "valid_base_n_massachussetts" is the same as "valid_clean".

## Models 

All building blocks are here. The two complete models are in Unet.py and DenseUnet.py. They can optionally have an attention mechanism built-in. Look at the documentation of each for more details.

#### Notebooks

We trained our models on colab using 3 notebooks:

* loss_experiment: Experiments on the loss, comparing which one performes the best 
* model_selection: Using the best loss, investigates which architecture has the best performance with the original dataset 
* train_best_arch: Training the two best architectures with the best loss on the original and "merged" dataset, using the pictures from the Massachussetts.

**Important note**: since all computations where done on colab, the saved models require a machine with a working cuda installation to work. Running `run.py` on a machine without cuda will issue in an error.


#### `metrics.py`

Used to compute the F1-score

#### `utils.py`

Contains methods that were used but not to define the model, nore to create the dataset. We find namely loss computations, loading of an image, combining of 256x256 images to predict larger ones and performance evaluation of the the model on validation data. 
