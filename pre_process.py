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

def circular_interpolate(series:pd.DataFrame, max_gap: int):
    is_na = series.isna()
    na_groups = (is_na != is_na.shift()).cumsum()
    gap_sizes = is_na.groupby(na_groups).transform('size')
    small_gaps = gap_sizes <= max_gap
    radians = np.deg2rad(series)
    
    x = np.cos(radians)
    y = np.sin(radians)

    x_interp = x.interpolate(method='linear')
    y_interp = y.interpolate(method='linear')

    interpolated_angle = np.rad2deg(np.arctan2(y_interp, x_interp)) % 360
    interpolated_series = series.where(~small_gaps, interpolated_angle)

    return interpolated_series, x_interp, y_interp

def winsorize_iqr(df, features, threshold=1.5):
    cleaned_df = df.copy()
    for feature in features:
        Q1 = cleaned_df[feature].quantile(0.25)
        Q3 = cleaned_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # 计算超出边界的值的数量
        lower_outliers = (cleaned_df[feature] < lower_bound).sum()
        upper_outliers = (cleaned_df[feature] > upper_bound).sum()
        
        # 截断处理
        cleaned_df[feature] = cleaned_df[feature].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"Winsorized {lower_outliers} lower and {upper_outliers} upper outliers from {feature}")
    return cleaned_df

if __name__ == "__main__":
    df = pd.read_csv('data.csv', na_values=['NA'])
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
    df['wd'] = df['wd'].map(direction_map)

    processed_df = df.copy()
    processed_df = winsorize_iqr(processed_df, features)
    processed_df['wd'], x_components, y_components = circular_interpolate(processed_df['wd'], LARGE_GAP_THRESHOLD)
    processed_df['wd_sin'] = y_components
    processed_df['wd_cos'] = x_components

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

    processed_df.to_csv('processed_data.csv', index=False)

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

