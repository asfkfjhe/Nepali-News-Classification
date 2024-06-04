from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from torchtext import data
import time
import torch.nn
from LSTM import RNN
from GRU import GRU
from nepali_stemmer.stemmer import NepStemmer
import nltk
from nltk.corpus import stopwords
import re

# Define the FastAPI app
app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define the input data schema
class TextRequest(BaseModel):
    text: str

GRU_MODEL_PATH = "GRU.pth"
LSTM_MODEL_PATH = "RNN.pth"

RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

VOCABULARY_SIZE = 60_000
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 50
DROPOUT = 0.5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIM = 200
BIDIRECTIONAL = True
HIDDEN_DIM = 200
NUM_LAYERS = 2
OUTPUT_DIM = 8

TEXT = data.Field(sequential=True,include_lengths=True) # necessary for packed_padded_sequence

LABEL = data.LabelField(dtype=torch.float)

fields = [('classlabel', LABEL), ('content', TEXT)]

train_dataset = data.TabularDataset(
    path="train.csv", format='csv',
    skip_header=True, fields=fields)

test_dataset = data.TabularDataset(
    path="test.csv", format='csv',
    skip_header=True, fields=fields)

valid_dataset = data.TabularDataset(
    path="valid.csv", format='csv',
    skip_header=True, fields=fields)

TEXT.build_vocab(train_dataset, test_dataset, valid_dataset,
                 min_freq=2)
LABEL.build_vocab(train_dataset)

INPUT_DIM_LSTM = 5422
INPUT_DIM_GRU = 5202

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

def clean_and_tokenize(text):
    #removing unnecessary symbols
    clean_text = re.sub(r'[\n,|।\'":]', '', text)

    #Tokenizing Text 
    nepstem = NepStemmer()
    tokenized_text = nepstem.stem(clean_text)

    #removing stopwords
    nep_stopwords = set(stopwords.words('nepali'))
    words = tokenized_text.split()
    filtered_words = [word for word in words if word.lower() not in nep_stopwords]
    C_T_S = ' '.join(filtered_words)

    return C_T_S

cats = {
    'politics' : 0, 
    'sport' : 1, 
    'entertainment' : 2, 
    'tech' : 3, 
    'business': 4
}

map_dict = {v: k for k, v in cats.items()}

# Load the trained GRU model
gru_model = GRU(INPUT_DIM_GRU, EMBEDDING_DIM, BIDIRECTIONAL, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT, PAD_IDX)
gru_model.load_state_dict(torch.load(GRU_MODEL_PATH))
gru_model.eval()

# Load the trained LSTM model
lstm_model = RNN(INPUT_DIM_LSTM, EMBEDDING_DIM, BIDIRECTIONAL, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT, PAD_IDX)
lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH))
lstm_model.eval()

# Define the predict function for both models
def predict(model, sentence):
    model.eval()
    indexed = [TEXT.vocab.stoi[token] for token in clean_and_tokenize(sentence).split()]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    
    with torch.no_grad():
        output = model(tensor, length_tensor)
        predictions = F.softmax(output, dim=1)
        max_prob, predicted_idx = torch.max(predictions, dim=1)
        predicted_label = map_dict[predicted_idx.item()]
        return {'category': predicted_label, 'probability': max_prob.item()}

# Define the API endpoint
@app.post("/predict")
async def classify_text(request: TextRequest):
    try:
        gru_prediction = predict(gru_model, request.text)
        lstm_prediction = predict(lstm_model, request.text)
        return {'gru_prediction': gru_prediction, 'lstm_prediction': lstm_prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve the main HTML file
@app.get("/")
async def read_index():
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    import nest_asyncio
    import uvicorn

    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)
