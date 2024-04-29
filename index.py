import pandas as pd 
import numpy as np 
dataset = pd.read_csv('/home/supriya/Desktop/FDV/Nepali-News-Classification/Nepali_Dataset_New.csv')
dataset['Category'].unique()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
dataset['Category']= label_encoder.fit_transform(dataset['Category'])
import re

def clean_text(string):
    clean_text = re.sub(r'[\n,|।\'":]', '', string)
    return clean_text

dataset['News'] = dataset['News'].apply(clean_text)
from nepalitokenizers import WordPiece

tokenizer = WordPiece()

for text in dataset['News']:
    tokens = tokenizer.encode(text)

print(tokens.ids)
print(tokens.tokens)



