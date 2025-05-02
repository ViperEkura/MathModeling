import torch
import pandas as pd
import torch.nn as nn


from torch import Tensor
from torch.utils.data import Dataset


class AirQualityDataset(Dataset):
    def __init__(self, file_path, feature_columns, target_column):
        self.data = pd.read_csv(file_path)
        self.features = self.data[feature_columns].values
        self.targets = self.data[target_column].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回单个样本
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return feature, target


class LSTM_predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM_predictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: Tensor):
        x, _ = self.lstm(x)
        x = self.norm(x)
        x = self.fc(x)
        return x
    

class Transformer_predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Transformer_predictor, self).__init__()
        self.attention = nn.Transformer(
            d_model=hidden_size, 
            dim_feedforward=4*hidden_size, 
            batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    
    def forward(self, x: Tensor):
        x = self.attention(x)
        x = self.norm(x)
        x = self.fc(x)
        return x