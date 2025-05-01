import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('processed_data.csv', na_values=['NA', 'NaN', 'nan'])

selected_columns = [
    'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
    'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM'
]

available_columns = [col for col in selected_columns if col in df.columns]
numeric_df = df[available_columns].copy()

for col in numeric_df.columns:
    numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

numeric_df.dropna(axis=1, how='all', inplace=True)

# 计算皮尔逊相关系数矩阵
corr_matrix = numeric_df.corr()

#  绘制热力图（仅显示上三角部分）
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
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

# 选择几个高度相关或感兴趣的变量对
interesting_pairs = [
    ('PM2.5', 'PM10'),
    ('SO2', 'NO2'),
    ('TEMP', 'O3'),
    ('TEMP', 'DEWP'),
    ('PRES', 'DEWP')
]

# 为每对变量绘制散点图
for x, y in interesting_pairs:
    if x in numeric_df.columns and y in numeric_df.columns:
        sns.jointplot(data=numeric_df, x=x, y=y, kind='reg', 
                     joint_kws={'scatter_kws': {'alpha': 0.5}})
        plt.suptitle(f"{x} vs {y}", y=1.02)
        plt.tight_layout()
        plt.savefig(f'scatter_{x}_vs_{y}.png')
        plt.show()