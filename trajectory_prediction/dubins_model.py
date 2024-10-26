import torch
import torch.nn as nn  # Import torch.nn for defining the RNN model
from torch.utils.data import Dataset


# Dataset class
class DubinsDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]

# RNN model
class DubinsRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(DubinsRNN, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.4)  # Dropout layer
        self.fc = nn.Linear(hidden_size, output_size)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize first hidden and cell states to 0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    
        LSTM_out, _ = self.LSTM(x, (h0, c0))
        # out = self.fc(rnn_out[:, -1, :])
        # LSTM_out[:, 0, :] = 0
        out = self.fc(LSTM_out)
        return out