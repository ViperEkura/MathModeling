import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW


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



class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Predictor, self).__init__()
        self.norm = nn.LayerNorm(input_size)
        self.fc = nn.LSTM(input_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: Tensor):
        x = self.norm(x)
        x = self.fc(x)
        x = x * F.sigmoid(self.gate(x))
        x = self.out(x)
        return x


def train_model(
    model: nn.Module, 
    dataset: Dataset, 
    batch_size=32, 
    epochs=10, 
    learning_rate=0.001, 
    device='cuda',
    save_plot_path='training_loss.png',
    test_split=0.2  # 新增参数，用于指定测试集比例
):
    # 划分训练集和测试集
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
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
    
    # 绘制训练和测试损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_losses, marker='o', linestyle='-', color='b', label='Train Loss')
    plt.plot(range(1, epochs+1), test_losses, marker='o', linestyle='-', color='r', label='Test Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    plt.savefig(save_plot_path)
    plt.close()


def predict(model: nn.Module, dataset: Dataset, batch_size=32, device='cuda'):
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []

    with torch.no_grad():
        for features, _ in dataloader:
            features = features.to(device)



def main():
    input_features = ['PM10','SO2', 'NO2', 'CO', 'O3', 'TEMP'
                      , 'PRES', 'RAIN', 'WSPM']
    target_feature = ['PM2.5']
    
    epochs = 20
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Predictor(
        input_size=len(input_features), 
        hidden_size=len(input_features) * 4, 
        output_size=len(target_feature), 
    )

    dataset = AirQualityDataset(
        'processed_data.csv', 
        feature_columns=input_features, 
        target_column=target_feature
    )
    train_model(model, dataset, batch_size=batch_size, epochs=epochs, device=device)

if __name__ == '__main__':
    main()