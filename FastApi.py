from fastapi import FastAPI, HTTPException
import torch
from torch.nn.functional import softmax
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
import spacy
import numpy as np
import uvicorn
import torch.nn as nn

# Define your news categories
cats = {
    'politics' : 0, 
    'sport' : 1, 
    'entertainment' : 2, 
    'tech' : 3, 
    'business': 4
}

# FastAPI instance
app = FastAPI()

# Model path
model_path = "RNN.pth"

# Load the model
try:
    model = torch.load(model_path, map_location=torch.device('cpu'))  # Load the model
    model.eval() 
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Tokenizer
tokenizer = get_tokenizer("basic_english")
# Assuming you have defined TEXT.vocab and other necessary variables
vocab = TEXT.vocab

# Function to preprocess text
def preprocess_text(text):
    tokenized_text = tokenizer(text)
    indexed_tokens = [vocab.stoi[token] for token in tokenized_text]
    length = len(indexed_tokens)
    return torch.tensor(indexed_tokens).unsqueeze(0), torch.tensor([length])

# Route for predicting news category
@app.post("/predict/")
async def predict_news_category(text: str):
    try:
        # Preprocess the text
        indexed_tokens, length = preprocess_text(text)

        # Perform prediction
        with torch.no_grad():
            output = model(indexed_tokens, length)
            probabilities = softmax(output, dim=1)
            predicted_class_index = torch.argmax(probabilities, dim=1).item()
            predicted_class = list(cats.keys())[list(cats.values()).index(predicted_class_index)]
        
        return {"predicted_category": predicted_class, "probabilities": probabilities.tolist()}
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error predicting: {str(e)}")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)
