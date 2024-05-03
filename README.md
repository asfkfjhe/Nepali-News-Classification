# Nepali News Classification

## Overview
This repository contains code for classifying Nepali news articles into different categories such as politics, sports, entertainment, tech, and business. The classification is done using various deep learning algorithms like LSTM and GRU.

## Dataset
The dataset used for this project consists of Nepali news articles collected from various online sources. Each news article is labeled with one of the predefined categories.

Categories:
- politics
- sports
- entertainment
- tech
- business

## Preprocessing
Before feeding the data into the models, the following preprocessing steps were performed:
- Cleaning : Removing unnecessary symbols
- Tokenization : using NepaliStemmer()
- Padding: Ensuring that all sequences have the same length by padding shorter sequences with zeros.
- Word Embedding: Converting words into dense vectors using pre-trained word embeddings like Word2Vec or GloVe.

## Models
### LSTM
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture that is capable of learning long-term dependencies. In this project, an LSTM model was trained on the preprocessed news articles to classify them into different categories.

### GRU
Gated Recurrent Unit (GRU) is another type of recurrent neural network (RNN) architecture that is similar to LSTM but with fewer parameters. A GRU model was also trained on the preprocessed data for comparison with the LSTM model.

## Evaluation
The accuracy of each model was evaluated using a separate test dataset that was not seen during training. The evaluation metrics used include accuracy, precision, recall, and F1-score for each category.

## Results
The results of the evaluation will be presented in the form of confusion matrices and classification reports for each model. These will help in understanding how well each model performs in classifying Nepali news articles into different categories.

## Conclusion
Based on the evaluation results, conclusions will be drawn regarding the effectiveness of LSTM and GRU models for Nepali news classification. Suggestions for further improvements and areas of future research will also be discussed.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Usage
1. Clone this repository:
