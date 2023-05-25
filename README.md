# Image Captioning - Group 2 - UAB Deep Learning Course
Our main proposal for this project is to be able to train an image captioning model based on the [Flickr 8k Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k).

*Click on Image to open Kaggle Dataset page:* 

[<img src="docs/site-logo.svg" width="120"/>](https://www.kaggle.com/datasets/adityajn105/flickr8k)

## General Project Structure
***Maybe we can omit this paragpraph, since we explain this on the Code structure?***

## Model architecture
Our model architecture is a combination of an encoder CNN network and a decoder LSTM network of two layers. 

For our CNN encoder we have used the pytorch's pretrained [ResNet-50 model](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html). The softmax from the last layer has been deleted and changed to a fully connected layer that runs through a linear function that feeds the embedding, to afterwards run through the first LSTM input.

For our embedding, we have used the 100 dimensions [GloVe pretrained embedding](https://nlp.stanford.edu/projects/glove/). Once the CNN output runs through the embedding, it runs directly into the first LSTM input. We have used a teacher forcing ratio of 0.5 to input each caption's word to the following LSTM cells.

## How to run the code
* As external files, you will need:
    * The *data* folder, which contains the *Flickr8k* dataset, formed by a folder with 8,091 images, and a .txt with 5 captions for each image.
    * The *GloVe Embedding* folder, which contains the **.pkl** files and the data files in order to train the LSTM decoder with a pretrained embedding.
        * It is recommended to download the data from [the GloVe official site](https://nlp.stanford.edu/projects/glove/). *We will be using the Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB):* [glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip)

Once you have this data, you will need to replicate the environment we used to develop the model. On the *environment.yml* file you will find all libraries needed to imitate our model. We will use *pytorch* as our deep learning framework, and we highly recommended to use CUDA platform to use the GPU potential in order to train the model.

## Code Structure
### General files:
* The *README.md* is this file which explains the repository information.
* The *environment.yml* file contains all the python dependencies that are needed to replicate our model experiments and to execute all scripts.
### Folders:
* The *data* folder contains the images and captions that will be used to feed the model.
* The *glove_files* folder will contain all the data that, after being processed, will be used to fit the embedding to the model's decoder.
* The *model* folder contains *model.py*, which is the described architecture for our CNN to LSTM model.
* The *utils* folder contains all functions that we developed in order to run our whole project. There we can find both metrics and generative functions.
### Main scripts:
* The *get_loader.py* script contains the *Vocabulary* and *Dataset* classes for our own project, with some dependency functions that are used to handle the training.
* The *test.py* script contains the testing function, which will be used to obtain each model's metrics and results.
* The *train_and_val.py* script contains the training and validation functions that will be used to train and evaluate our model during the training phase, no pun intended. It also contains a training function that depicts some images from the training batch and shows the current epoch's caption prediction. 
* The *training_model.ipynb* notebook contains the training of each of our tried models and its results. This way, we keep all training stored in a same file. 

## Further improvements
We have also trained the model with a much wider dataset, this time containing 30,000 images in total, in order to avoid overfitting and have a more general model. To do so, we used the [Flickr30k dataset](https://www.kaggle.com/datasets/eeshawn/flickr30k), which keeps the same structure as the 8k, but with more information. This way, we do not have to focus on adapting the dataset to our own Dataset and Vocabulary classes.

## Example Code
***THIS PART CAN BE OMITTED, YOU CAN CHOOSE***

The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate imgcaption
```

To run the example code:
```
python main.py
```
***THIS PART CAN BE OMITTED, YOU CAN CHOOSE***

## Contributors
* **Miguel Moral Hernández - miguel.moral@autonoma.cat**
* **Àlex Sànchez Zurita - alex.sanchezz@autonoma.cat**
* **Pol Medina Arévalo - pol.medina@autonoma.cat**


### ***Neural Networks and Deep Learning Course***

### ***Artificial Intelligence Degree***

### ***UAB 2023***
