import torch
import torch.nn as nn

class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128, num_layers=2, dropout=0.2):
        super(SimpleRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        out, hidden = self.rnn(embedded, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden

class BiRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128, num_layers=2, dropout=0.2):
        super(BiRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        out, hidden = self.rnn(embedded, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        out, hidden = self.lstm(embedded, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        out, hidden = self.gru(embedded, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden
