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
sns.set_theme(font_scale=1.2)
plt.rcParams['axes.titlesize'] = 16 
plt.rcParams['axes.labelsize'] = 14 

plt.figure(figsize=(32, 24))
pair_plot = sns.pairplot(
    numeric_df[selected_columns],
    kind='reg',
    diag_kind='kde',
    plot_kws={
        'scatter_kws': {'alpha': 0.6, 's': 20}, 
        'line_kws': {'color': 'purple'}  
    },
    diag_kws={'fill': True}
)

pair_plot.figure.suptitle("Pairwise Relationships of Air Quality Features", y=1.02)
plt.tight_layout()
plt.savefig('pair_matrix.png', bbox_inches='tight')