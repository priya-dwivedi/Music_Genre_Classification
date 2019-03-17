This repository contains the code for Music Genre Recognition project

## Packages
* Python 3.6.5
* Tensorflow - 1.7.0
* Keras - 2.2.4
* Numpy, Pandas, Matplotlib
* Librosa - 0.6.2

## Raw data:
Download FMA Small from: https://github.com/mdeff/fma
Raw data is 8GB and consists of audio from 8000 songs + metadata with features like MFCC

## Preprocessed data
The raw audio has been converted to mel-spectograms and pickled. There are 3 files for training, validation and testing on the drive link -https://drive.google.com/drive/u/0/folders/1-PTQBiz6E53uUa9LebHjds_ZQesRHEqx

You only need these files for running any of the notebooks with the neural networks. 

## Code Notebooks

#### Explore data, convert raw audio into spectograms and pickle them
To run code in any of these notebooks, first please download raw data from FMA Github link above 
1. load_fma_dataset: Loads fma_dataset and explores it. 
2. Plot_Spectograms: Plots spectograms for the 8 different genres
3. convert_to_npz: Loads the raw audio, converts each file to a spectogram and pickles the results to make it easy for training models. The output from this are the datasets in the drive link above

### Building models
To run the code below, please download the processed data from the drive
1. baseline_model_fma: This model uses the metadata in tracks.csv to load MFCC features and builds a SVC classifier.

2. CRNN_model: This notebook uses the compressed spectograms to build a CRNN model in Keras

3. CNN_RNN_Parallel: This notebook uses the compressed spectograms to build a a parallel CNN-RNN model in Keras

4. models folder has the trained weights for the 2 models. 

### Activation Visualization and Embedding Clustering

1. Activation_Visualization: This notebook loads the weights for Parallel CNN-RNN model and uses the keras_vis package to draw activation visualizations for the filters in convolution block 1 and convolution block 5

2. Embedding_Clustering_CRNN: This notebook extracts the features from the first dense layer of CRNN model and performs clustering on them. It then compares the outputs of the clustering with the true labels

