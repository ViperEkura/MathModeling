# modeling.py

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
import random
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(3304)


class AirQualityDataset(Dataset):
    def __init__(self, file_path, feature_columns, target_column, training):
        self.data = pd.read_csv(file_path)
        if training:
            index = (self.data["year"] >= 2013) & (self.data["year"] <= 2015)
            self.data = self.data[index]
        else:
            index = self.data["year"] >= 2016
            self.data = self.data[index]

        self.data = self.data.dropna()
        self.features = self.data[feature_columns].values
        self.targets = self.data[target_column].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return feature, target


class StaticCondition(Dataset):
    def __init__(self, file_path, feature_columns, target_column):
        self.data = pd.read_csv(file_path)
        self.data = self.data[self.data['WSPM'] < 1.0].copy() 
        self.data.dropna(inplace=True)

        self.features = self.data[feature_columns].values
        self.targets = self.data[target_column].values

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return feature, target

    def __len__(self):
        return len(self.data)


class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Predictor, self).__init__()
        self.norm_in = nn.BatchNorm1d(input_size)
        self.fc = nn.Linear(input_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.norm_out = nn.BatchNorm1d(hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor):
        x = self.norm_in(x)
        x = self.fc(x)
        x = x * F.sigmoid(self.gate(x))
        x = self.norm_out(x)
        x = self.out(x)
        return x


def r2_score_func(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_mean = torch.mean(y_true)
    sst = torch.sum((y_true - y_mean) ** 2)
    sse = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - sse / sst
    return r2


def mae_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    return torch.mean(torch.abs(y_true - y_pred))


def main(): 
    input_features = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP',
                      'PRES', 'RAIN', 'WSPM', 'wd_cos', 'wd_sin']
    target_feature = ['PM2.5']

    epochs = 15
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 0.001
    save_plot_path = 'training_loss.png'

    model = Predictor(
        input_size=len(input_features),
        hidden_size=len(input_features) * 4,
        output_size=len(target_feature),
    )

    train_dataset = AirQualityDataset(
        'processed_data.csv',
        feature_columns=input_features,
        target_column=target_feature,
        training=True,
    )
    test_dataset = AirQualityDataset(
        'processed_data.csv',
        feature_columns=input_features,
        target_column=target_feature,
        training=False
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 初始化记录损失的列表
    train_losses = []
    test_losses = []

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for (features, targets) in train_dataloader:
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(features)
            loss = F.mse_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # 测试阶段
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for features, targets in test_dataloader:
                features = features.to(device)
                targets = targets.to(device)

                outputs = model(features)
                loss = F.mse_loss(outputs, targets)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_dataloader)
        test_losses.append(avg_test_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-', color='b', label='Train Loss')
    plt.plot(range(1, epochs + 1), test_losses, marker='o', linestyle='-', color='r', label='Test Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_plot_path)
    plt.close()

    model.eval()
    test_data = test_dataset.data.copy()
    all_features = torch.tensor(test_dataset.features, dtype=torch.float32).to(device)

    with torch.no_grad():
        test_data['Predicted_PM2_5'] = model(all_features).cpu().numpy()

    y_true = torch.tensor(test_data['PM2.5'].values, dtype=torch.float32)
    y_pred = torch.tensor(test_data['Predicted_PM2_5'].values, dtype=torch.float32)

    mse = F.mse_loss(y_pred, y_true).item()
    r2_score = r2_score_func(y_true, y_pred).item()
    mae = mae_loss(y_true, y_pred).item()
    
    print("\nEvaluation Metrics on Test Set:")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2_score:.4f}")
    print(f"MAE: {mae:.4f}")

    test_data.to_csv("test_predictions.csv", index=False)
    torch.save(model.state_dict(), 'model.pt')

    static_dataset = StaticCondition(
        file_path='processed_data.csv',
        feature_columns=input_features,
        target_column=target_feature
    )

    static_loader = DataLoader(static_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for features, targets in static_loader:
            features = features.to(device)
            outputs = model(features).squeeze()

            all_preds.append(outputs.cpu())
            all_trues.append(targets)

    all_preds = torch.cat(all_preds, dim=0).flatten()
    all_trues = torch.cat(all_trues, dim=0).flatten()

    mse = F.mse_loss(all_preds, all_trues).item()
    r2 = r2_score_func(all_trues, all_preds).item()
    mae = mae_loss(all_preds, all_trues).item()

    print(f"\nStable Weather Evaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"MAE: {mae:.4f}")


if __name__ == '__main__':
    main()