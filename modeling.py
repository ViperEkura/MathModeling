import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam


class AirQualityDataset(Dataset):
    def __init__(self, file_path, feature_columns, target_column):
        self.data = pd.read_csv(file_path)
        self.data.dropna(inplace=True)
        self.features = self.data[feature_columns].values
        self.targets = self.data[target_column].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return feature, target


class LSTM_predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM_predictor, self).__init__()
        self.norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size,
            num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: Tensor):
        x = self.norm(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
    

class Transformer_predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=8, dropout=0.1):
        super(Transformer_predictor, self).__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, 
            num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        res = self.input_projection(x)
        x = self.norm(res)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output + res
        return attn_output
    

def train_model(
    model:nn.Module, 
    dataset: Dataset, 
    batch_size=32, 
    epochs=10, 
    learning_rate=0.001, 
    device='cuda'
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    for epoch in range(epochs):
        model.train() 
        total_loss = 0

        for batch_idx, (features, targets) in enumerate(dataloader):
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(features)
            loss = F.mse_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}],"
                      " Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")


def main():
    model = LSTM_predictor(input_size=4, hidden_size=64, output_size=1, num_layers=2)
    dataset = AirQualityDataset(
        'processed_data.csv', 
        feature_columns=['PM10', 'SO2', 'NO2', 'CO'], 
        target_column=['PM2.5'])
    train_model(model, dataset)

if __name__ == '__main__':
    main()