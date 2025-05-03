import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # 新增导入

if __name__ == '__main__':
    df = pd.read_csv('processed_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values(by='datetime', inplace=True)

    # 提取时间特征
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year

    features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

    time_periods = {
        "Hourly": ("hour", 'Hourly'),
        "Monthly": ("month", 'Monthly'), 
        "Yearly": ("year", 'Yearly')
    }

    for period, (group_key, suffix) in time_periods.items():
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        fig.suptitle(f'{period} Average of Features', fontsize=16)
        
        for i, feature in enumerate(features):
            if period == 'Yearly':
                df = df[df['datetime'].dt.year != 2017]
                
            avg_value = df.groupby(group_key)[feature].mean()
            ax = axes[i // 3, i % 3]
            

            ax.plot(avg_value.index, avg_value.values, 
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    color='royalblue',
                    markersize=6,
                    markerfacecolor='red')
            
            ax.set_title(f'{feature} - {suffix}', fontsize=12, pad=10)
            
            xlabel_map = {
                'hour': "Hour of Day",
                'month': "Month",
                'year': "Year"
            }
            ax.set_xlabel(xlabel_map.get(group_key, group_key), fontsize=10)
            ax.set_ylabel("Average Value", fontsize=10)
            
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            if group_key == 'hour':
                ax.set_xticks(range(0, 24, 2))
            elif group_key == 'month':
                ax.set_xticks(range(1, 13))

            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_axisbelow(True)
                
        plt.tight_layout(pad=3.0)
        plt.savefig(f'average_features_{period}.png', dpi=150, bbox_inches='tight')
        plt.close()