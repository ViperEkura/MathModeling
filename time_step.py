import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('processed_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.sort_values(by='datetime', inplace=True)

df['hour'] = df['datetime'].dt.hour
df['day_of_year'] = df['datetime'].dt.dayofyear
df['cycle_120'] = (df['day_of_year'] - 1) // 120 + 1
df['year'] = df['datetime'].dt.year

features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

time_periods = {
    "Hourly": ("hour", '24h'),
    "Cycle_120d": ("cycle_120", '120d'),
    "Yearly": ("year", 'Yearly')
}

for period, (group_key, suffix) in time_periods.items():
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'{period} Average of Features', fontsize=16)
    
    for i, feature in enumerate(features):
        avg_value = df.groupby(group_key)[feature].mean()
        ax = axes[i // 4, i % 4]
        ax.plot(avg_value.index, avg_value.values, marker='o', markersize=4)
        ax.set_title(f'{feature} - {suffix}')
        ax.set_xlabel(group_key if group_key != 'hour' else "Hour of Day")
        ax.set_ylabel("Average Value")
        ax.grid(True)
    
    # 如果特征数量不是子图总数的倍数，则删除多余的子图
    for j in range(len(features), 2 * 4):
        fig.delaxes(axes[j // 4, j % 4])
        
    plt.tight_layout()
    plt.savefig(f'average_features_{period}.png', dpi=150, bbox_inches='tight')