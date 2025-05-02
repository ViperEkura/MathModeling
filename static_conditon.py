import torch
import torch.nn.functional as F
import pandas as pd

from modeling import Predictor, r2_score_func
from torch.utils.data import Dataset, DataLoader


# 给出静态稳定条件下的计算方法
class StaticCondition(Dataset):
    def __init__(self, file_path, feature_columns, target_column):
        self.data = pd.read_csv(file_path)
        self.data = self.data[self.data['WSPM'] < 1.0]
        self.data.dropna(inplace=True)
        print(self.data)

        self.features = self.data[feature_columns].values
        self.targets = self.data[target_column].values

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def __len__(self):
        return len(self.data)
    


def main():
    model_path = 'model.pt'
    input_features = ['PM10','SO2', 'NO2', 'CO', 'O3', 'TEMP',
                      'PRES', 'RAIN', 'WSPM', 'wd_cos', 'wd_sin']
    target_feature = ['PM2.5']

    # 加载模型
    state_dict = torch.load(model_path, weights_only=True)
    model = Predictor(
        input_size=len(input_features), 
        hidden_size=len(input_features) * 4, 
        output_size=len(target_feature), 
    )
    model.load_state_dict(state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device, dtype=torch.float64)
    model.eval()

    low_wind_dataset = StaticCondition(
        file_path='processed_data.csv',
        feature_columns=input_features,
        target_column=target_feature
    )


    low_wind_loader = DataLoader(low_wind_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for features, targets in low_wind_loader:
            features = features.to(device)
            outputs = model(features).squeeze()

            all_preds.append(outputs.cpu())
            all_trues.append(targets)

    # 合并所有 batch
    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)

    # 计算指标
    mse = F.mse_loss(all_preds, all_trues).item()
    r2 = r2_score_func(all_trues, all_preds)
    
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
    

if __name__ == '__main__':
    main()
    
    