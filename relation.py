import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('processed_data.csv', na_values=['NA', 'NaN', 'nan'])

selected_columns = [
    'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
    'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'wd'
]

available_columns = [col for col in selected_columns if col in df.columns]
numeric_df = df[available_columns].copy()

for col in numeric_df.columns:
    numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

numeric_df.dropna(axis=1, how='all', inplace=True)

# 计算皮尔逊相关系数矩阵
corr_matrix = numeric_df.corr()
# 创建一个上三角掩码（mask）
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

#  绘制热力图（仅显示上三角部分）
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    mask=mask,              # 应用上三角掩码
    annot=True,             # 显示相关系数数值
    fmt=".2f",              # 数值保留两位小数
    cmap='coolwarm',        # 颜色映射方案
    linewidths=0.5,         # 单元格间距
    square=True,            # 单元格为正方形
    cbar_kws={"shrink": .8} # 缩短颜色条长度
)
plt.title("Upper Triangle of Correlation Matrix")
plt.tight_layout()
plt.savefig('correlation_matrix.png')