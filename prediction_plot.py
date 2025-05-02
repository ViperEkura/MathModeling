import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    df = pd.read_csv('test_predictions.csv')
    df = df[df['year'] == 2016]

    y_true_np = df['PM2.5'].values
    y_pred_np = df['Predicted_PM2_5'].values
    

    if 'datetime' in df.columns:
        time = pd.to_datetime(df['datetime'])
    else:
        time = np.arange(len(y_true_np)) 

    plt.figure(figsize=(28, 6))

    plt.plot(time, y_true_np, label='True PM2.5', color='#1f77b4', linewidth=2, alpha=0.8)
    plt.plot(time, y_pred_np, label='Predicted PM2.5', color='#ff7f0e', linestyle='--', linewidth=2)

    if 'WSPM' in df.columns:
        calm_periods = df['WSPM'] < 0.5 
        plt.scatter(time[calm_periods], y_true_np[calm_periods], 
                    color='red', s=20, label='Calm Wind (WSPM < 0.5 m/s)', alpha=0.6)

    plt.xlabel('Time', fontsize=12)
    plt.ylabel('PM2.5 (µg/m³)', fontsize=12)
    plt.title('True vs Predicted PM2.5 Concentration', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, framealpha=1)

    if 'datetime' in df.columns:
        plt.gcf().autofmt_xdate()

    plt.savefig('pm25_true_vs_pred.png', dpi=300, bbox_inches='tight')
    
if __name__ == "__main__":
    main()