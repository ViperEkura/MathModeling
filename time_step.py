import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('processed_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.sort_values(by='datetime', inplace=True)

# 提取时间特征
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

time_periods = {
    "Hourly": ("hour", '24h'),
    "Monthly": ("month", 'Monthly'), 
    "Yearly": ("year", 'Yearly')
}

for period, (group_key, suffix) in time_periods.items():
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'{period} Average of Features', fontsize=16)
    
    for i, feature in enumerate(features):
        avg_value = df.groupby(group_key)[feature].mean()
        ax = axes[i // 4, i % 4]
        
        # 修改为柱状图，并调整样式
        ax.bar(avg_value.index, avg_value.values, color='skyblue', edgecolor='black')
        
        ax.set_title(f'{feature} - {suffix}')
        
        # 设置坐标轴标签
        xlabel_map = {
            'hour': "Hour of Day",
            'month': "Month",
            'year': "Year"
        }
        ax.set_xlabel(xlabel_map.get(group_key, group_key))
        ax.set_ylabel("Average Value")
        ax.grid(True, axis='y')  # 仅显示水平网格线

    # 清理多余子图
    for j in range(len(features), 2 * 4):
        fig.delaxes(axes[j // 4, j % 4])
        
    plt.tight_layout()
    plt.savefig(f'average_features_{period}.png', dpi=150, bbox_inches='tight')