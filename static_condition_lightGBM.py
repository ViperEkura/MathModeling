from modeling_lightGBM import AirQualityModel 
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import torch

def mae_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_true - y_pred))


def evaluate_stable_weather():
    model = AirQualityModel()
 
    train_data, _ = model.prepare_data('processed_data.csv')
    model.train(train_data)

    data = pd.read_csv('processed_data.csv')
    stable_weather_data = data[data['WSPM'] < 1.0] 
    stable_weather_data.dropna(inplace=True)


    X_stable = stable_weather_data[model.feature_columns]
    y_stable = stable_weather_data[model.target_column]

    predictions = model.model.predict(X_stable)

    mse = mean_squared_error(y_stable, predictions)
    r2 = r2_score(y_stable, predictions)

    y_true_tensor = torch.tensor(y_stable.values, dtype=torch.float32)
    y_pred_tensor = torch.tensor(predictions, dtype=torch.float32)

    mae = torch.nn.functional.l1_loss(y_true_tensor, y_pred_tensor).item()
    stable_weather_data['Predicted_PM2_5'] = predictions
    stable_weather_data.to_csv("stable_weather_predictions_lgbm.csv", index=False)


    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

if __name__ == '__main__':
    evaluate_stable_weather()