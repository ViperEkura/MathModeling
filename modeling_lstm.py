import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


class AirQualityDataset(Dataset):
    def __init__(self, file_path, feature_columns, target_column, training, seq_len=1):
        self.data = pd.read_csv(file_path)
        self.seq_len = seq_len
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        if training:
            index = (self.data["year"] >= 2013) & (self.data["year"] <= 2015)
            self.data = self.data[index]
        else:
            index = self.data["year"] >= 2016
            self.data = self.data[index]
            
        self.data.dropna(inplace=True)
        
        # 创建序列数据
        self.features, self.targets = self.create_sequences()
        # 保存原始数据的子集（去掉前seq_len-1个无法形成完整序列的样本）
        self.valid_data = self.data.iloc[self.seq_len-1:].copy()

    def create_sequences(self):
        features = []
        targets = []
        
        data_values = self.data[self.feature_columns].values
        target_values = self.data[self.target_column].values
        
        for i in range(len(data_values) - self.seq_len + 1):  # 修改这里确保长度匹配
            features.append(data_values[i:i+self.seq_len])
            targets.append(target_values[i+self.seq_len-1])  # 预测序列最后一个时间点的值
        
        return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return feature, target


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层
        out = self.fc(out)
        return out
    
def r2_score_func(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_mean = torch.mean(y_true)
    sst = torch.sum((y_true - y_mean) ** 2)
    sse = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - sse / sst
    return r2.item()


def main():
    input_features = ['PM10','SO2', 'NO2', 'CO', 'O3', 'TEMP',
                      'PRES', 'RAIN', 'WSPM', 'wd_cos', 'wd_sin']
    target_feature = 'PM2.5'
    
    # 可调参数
    epochs = 15
    batch_size = 32
    seq_len = 7  # 使用7天的数据作为序列
    hidden_size = 64
    num_layers = 2  # LSTM层数
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 0.001
    save_plot_path = 'training_loss.png'
    
    model = LSTMPredictor(
        input_size=len(input_features),
        hidden_size=hidden_size,
        output_size=1,
        num_layers=num_layers
    )

    train_dataset = AirQualityDataset(
        'processed_data.csv', 
        feature_columns=input_features, 
        target_column=target_feature,
        training=True,
        seq_len=seq_len
    )
    test_dataset = AirQualityDataset(
        'processed_data.csv', 
        feature_columns=input_features, 
        target_column=target_feature,
        training=False,
        seq_len=seq_len
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    
    model.to(device)

    for epoch in range(epochs):
        model.train() 
        total_train_loss = 0

        for batch_idx, (features, targets) in enumerate(train_dataloader):
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1)  # 添加维度匹配输出形状

            outputs = model(features)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for features, targets in test_dataloader:
                features = features.to(device)
                targets = targets.to(device).unsqueeze(1)
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_dataloader)
        test_losses.append(avg_test_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    
    # 绘制损失曲线
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
    
    # 评估模型
    model.eval()
    all_features = torch.tensor(test_dataset.features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions = model(all_features).cpu().numpy().flatten()
    
    # 使用保存的valid_data确保长度匹配
    results = test_dataset.valid_data.copy()
    results['Predicted_PM2.5'] = predictions
    
    # 计算R2分数
    r2_score = r2_score_func(
        torch.tensor(results[target_feature].values, dtype=torch.float32),
        torch.tensor(results['Predicted_PM2.5'].values, dtype=torch.float32)
    )

    print(f"R2 Score: {r2_score:.4f}")
    results.to_csv("test_predictions.csv", index=False)

if __name__ == '__main__':
    main()