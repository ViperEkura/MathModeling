import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.read_csv('data.csv', na_values=['NA', 'NaN', 'nan'])
df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
df = df.sort_values('datetime').set_index('datetime')

interpolated_df = df.groupby('station', group_keys=False).apply(
    lambda group: group.interpolate(
        method='linear',         # 线性插值
        limit_direction='both',  # 双向填充
        limit=29                 # 最多填充连续3个缺失值
    )
)

plt.figure(figsize=(15, 6))

# 获取需要标记的插值区域（原始缺失但插值后存在的点）
mask = df['PM2.5'].isna() & interpolated_df['PM2.5'].notna()

# 生成插值区间标记
interp_ranges = (mask != mask.shift()).cumsum()[mask]
for _, group in interp_ranges.groupby(interp_ranges):
    start_time = group.index[0]
    end_time = group.index[-1]
    plt.axvspan(start_time, end_time, 
                facecolor='grey', alpha=0.2, 
                edgecolor='none', zorder=0)

# 绘制插值后的时序数据
interpolated_df['PM2.5'].plot(
    color='#2c7bb6',  # 海蓝色
    linewidth=1.2,
    label='Interpolated PM2.5'
)

# 绘制原始有效数据点
df['PM2.5'].dropna().plot(
    style='-',
    markersize=3,
    color='#d7191c',  # 警示红
    alpha=0.7,
    label='Original Data'
)

plt.title('PM2.5 Time Series with Linear Interpolation (Gray Background = Filled Regions)')
plt.xlabel('Datetime')
plt.ylabel('PM2.5 Concentration (μg/m³)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()