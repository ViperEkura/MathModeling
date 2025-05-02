import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


class AirQualityDataset(Dataset):
    def __init__(self, file_path, feature_columns, target_column, training):
        self.data = pd.read_csv(file_path)
        if training:
            index = (self.data["year"] >= 2013) & (self.data["year"] <= 2015)
            self.data = self.data[index]
        else:
            index = self.data["year"] >= 2016
            self.data = self.data[index]
            
        self.data.dropna(inplace=True)

        self.features = self.data[feature_columns].values
        self.targets = self.data[target_column].values
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return feature, target



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
        self.norm_out(x)
        x = self.out(x)
        return x
    
def r2_score_func(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_mean = torch.mean(y_true)
    sst = torch.sum((y_true - y_mean) ** 2)
    sse = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - sse / sst
    return r2.item()


def main():
    input_features = ['PM10','SO2', 'NO2', 'CO', 'O3', 'TEMP'
                      , 'PRES', 'RAIN', 'WSPM', 'wd_cos', 'wd_sin']
    target_feature = ['PM2.5']
    
    epochs = 15
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate=0.001
    device='cuda'
    save_plot_path='training_loss.png'
    
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

        # 训练阶段
        for batch_idx, (features, targets) in enumerate(train_dataloader):
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
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_losses, marker='o', linestyle='-', color='b', label='Train Loss')
    plt.plot(range(1, epochs+1), test_losses, marker='o', linestyle='-', color='r', label='Test Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_plot_path)
    plt.close()
    
    model.eval()
    all_data = test_dataset.data.copy()
    all_features = torch.tensor(test_dataset.features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        all_data['Predicted_PM2_5'] = model(all_features).cpu().numpy()

    r2_score = r2_score_func(
        torch.tensor(all_data['PM2.5'].values, dtype=torch.float32),
        torch.tensor(all_data['Predicted_PM2_5'].values, dtype=torch.float32)
    )

    print(f"R2 Score: {r2_score:.4f}")
    all_data.to_csv("test_predictions.csv", index=False)

if __name__ == '__main__':
    main()