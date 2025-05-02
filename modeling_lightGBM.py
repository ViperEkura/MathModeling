import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import train_test_split

class AirQualityModel:
    def __init__(self):
        self.model = None
        self.feature_columns = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 
                               'PRES', 'RAIN', 'WSPM', 'wd_cos', 'wd_sin']
        self.target_column = 'PM2.5'
        
    def prepare_data(self, file_path):
        # Load and split data
        data = pd.read_csv(file_path)
        
        # Training data (2013-2015)
        train_data = data[(data["year"] >= 2013) & (data["year"] <= 2015)]
        
        # Test data (2016+)
        test_data = data[data["year"] >= 2016]
        
        # Drop NA values
        train_data = train_data.dropna()
        test_data = test_data.dropna()
        
        return train_data, test_data
    
    def train(self, train_data, params=None):
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'mse',  # Changed from rmse to mse
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
        
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        
        test_data['Predicted_PM2_5'] = predictions
        
        return test_data, mse, r2
    
    def plot_feature_importance(self):
        if self.model is not None:
            lgb.plot_importance(self.model, importance_type='gain')
            plt.title('Feature Importance')
            plt.show()
        else:
            print("Model not trained yet.")

def main():
    # Initialize model
    model = AirQualityModel()
    
    # Prepare data
    train_data, test_data = model.prepare_data('processed_data.csv')
    
    # Train model
    model.train(train_data)
    
    # Evaluate model
    test_data_with_preds, mse, r2 = model.evaluate(test_data)
    
    # Save predictions
    test_data_with_preds.to_csv("test_predictions_lgbm.csv", index=False)
    
    # Plot feature importance
    model.plot_feature_importance()
    
    print(f"\nFinal Metrics: MSE={mse:.4f}, R²={r2:.4f}")

if __name__ == '__main__':
    main()