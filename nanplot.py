import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# 定义"大段缺失"的阈值（例如连续50小时以上）
LARGE_GAP_THRESHOLD = 50

features_without_wd = ['PM2.5','PM10','SO2', 'NO2', 'CO', 'O3', 'TEMP',
                       'PRES', 'DEWP', 'RAIN', 'WSPM']
features = features_without_wd + ['wd']
direction_map = {
    'N':0, 'NNE':22.5, 'NE':45, 'ENE':67.5,
    'E':90, 'ESE':112.5, 'SE':135, 'SSE':157.5,
    'S':180, 'SSW':202.5, 'SW':225, 'WSW':247.5,
    'W':270, 'WNW':292.5, 'NW':315, 'NNW':337.5
}

df = pd.read_csv('data.csv', na_values=['NA'])
df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
df['wd'] = df['wd'].map(direction_map)

#对风向映射进行插值处理



processed_df = df.copy()
for feature in features_without_wd:
    # 1. 标记连续缺失段
    is_na = processed_df[feature].isna()
    na_groups = (is_na != is_na.shift()).cumsum()
    gap_sizes = is_na.groupby(na_groups).transform('size')
    
    # 2. 只对小段缺失进行线性插值
    small_gaps = gap_sizes <= LARGE_GAP_THRESHOLD
    processed_df[feature] = processed_df[feature].where(
        ~small_gaps,  # 保留大段缺失为NaN
        processed_df[feature].interpolate(method='linear')  # 小段线性插值
    )



# 绘制处理前后的缺失值对比图
plt.figure(figsize=(15, 6))

# 原始缺失情况
plt.subplot(2, 1, 1)
sns.heatmap(df[features].isna().T, 
            cbar=False,
            cmap='viridis',
            yticklabels=True)
plt.title("Original Missing Values")
plt.xlabel("Data Records")
plt.ylabel("Features")  
plt.yticks(rotation=0)

# 处理后的缺失情况
plt.subplot(2, 1, 2)
sns.heatmap(processed_df[features].isna().T, 
            cbar=False,
            cmap='viridis',
            yticklabels=True)
plt.title(f"After Processing (Gaps > {LARGE_GAP_THRESHOLD} hours kept as NaN)")
plt.xlabel("Data Records")
plt.ylabel("Features")  
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('missing_values_comparison.png')
