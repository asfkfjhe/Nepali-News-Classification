import torch 
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_dim, embedding_dim, bidirectional, hidden_dim, num_layers, output_dim, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, 
                          hidden_dim,
                          num_layers=num_layers,
                          bidirectional=bidirectional, 
                          dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * num_layers, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_length):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_length)
        packed_output, hidden = self.gru(packed_embedded)
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        hidden = self.fc1(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        return hidden
