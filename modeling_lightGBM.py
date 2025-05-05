# modeling_lightGBM.py

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb

class AirQualityModel:
    def __init__(self):
        self.model = None
        self.feature_columns = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 
                               'PRES', 'RAIN', 'WSPM', 'wd_cos', 'wd_sin']
        self.target_column = 'PM2.5'
        
    def prepare_data(self, file_path):
        data = pd.read_csv(file_path)
        train_data = data[(data["year"] >= 2013) & (data["year"] <= 2015)]
        test_data = data[data["year"] >= 2016]
        
        train_data = train_data.dropna()
        test_data = test_data.dropna()
        
        return train_data, test_data
    
    def train(self, train_data, params=None):
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'mse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42
            }
        
        X_train = train_data[self.feature_columns]
        y_train = train_data[self.target_column]
        train_set = lgb.Dataset(X_train, label=y_train)
        
        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            callbacks=[lgb.log_evaluation(50)]
        )
    
    def evaluate(self, test_data):
        X_test = test_data[self.feature_columns]
        y_test = test_data[self.target_column]

        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        print("\nEvaluation Metrics on Test Set:")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"MAE: {mae:.4f}") 

        test_data['Predicted_PM2_5'] = predictions

        return test_data, mse, r2, mae
    
    def evaluate_stable_weather(self, file_path):
        data = pd.read_csv(file_path)
        stable_weather_data = data[data['WSPM'] < 1.0].copy()
        stable_weather_data.dropna(inplace=True)

        X_stable = stable_weather_data[self.feature_columns]
        y_stable = stable_weather_data[self.target_column]

        predictions = self.model.predict(X_stable)

        mse = mean_squared_error(y_stable, predictions)
        r2 = r2_score(y_stable, predictions)
        mae = mean_absolute_error(y_stable, predictions)

        print(f"\nStable Weather Evaluation (WSPM < 1.0):")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"MAE: {mae:.4f}")

        stable_weather_data['Predicted_PM2_5'] = predictions
        stable_weather_data.to_csv("stable_weather_predictions_lgbm.csv", index=False)

        return mse, r2, mae

def main():
    model = AirQualityModel()
    train_data, test_data = model.prepare_data('processed_data.csv')
    model.train(train_data)

    test_data_with_preds, _, _, _ = model.evaluate(test_data)
    model.evaluate_stable_weather('processed_data.csv')
    test_data_with_preds.to_csv("test_predictions_lgbm.csv", index=False)


if __name__ == '__main__':
    main() 