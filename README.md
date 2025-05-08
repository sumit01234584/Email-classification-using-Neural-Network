# Email-classification-using-Neural-Network

## This repository has all the code to train a model, infer a model and launch an application about to classify a text as spam or ham. We also have a preprocessing script to preprocess any raw email dataset.

## Installation
- Python 3.9
- flask
- pandas
- sklearn
- pytorch
- tqdm

## Folder Description
### Dataset:
- spam.csv: original dataset
- training_data.csv: data on which our model is trained on
- test_data.csv: data on which our model is tested on
- spam_ham_ratios.pkl: pickle file having ratios of spam to ham in our datasets
- word_colum_dict.pkl: word to number mapping file
- vocab.pkl: vocabulary file that on which entire model relies

### checkpoints: folder to save checkpoints of trained folder for each epoch so as to choose the best

### App: Folder to have the code to launch an application in browser, allowing user to enter any text in real time and classify it as spam or ham. Description is given below:
- checkpoints: folder to store the best performing model checkpoint
- dataset: keeping word_colum_dict file from previous dataset folder
- model.py: original model file
- app.py: flask application file, can edit html and css code to make the application more appealing

### Files:
- dataloader.py: file to make dataset in batches and load it into the model
- model.py: model architecture that is taken to train (pytorch based)
- preprocessing.py: file that preprocesses the original dataset (spam.csv) and splits it into train and test and then creates vocab and other files to be used by model
- train.py: script to train the model on training dataset, printing the metrics to measure the performance on testing dataset of the model along with saving checkpoints (to be run only if training again, no need to train again)
- inference.py: script to test the performance of any checkpoints

#### After training model for 50 epochs, we got the following performance on testing dataset at epoch 2, which was the best among all epochs:
| Accuracy | 98.80 \\% |
| Precision | 97.64 \\% |
| Recall | 93.23 \\% |
| F1 Score | 95.38 \\% |
