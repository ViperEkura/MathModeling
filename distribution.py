import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
df = pd.read_csv('data.csv', na_values=['NA', 'NaN', 'nan'])

# 2. 选择数值型字段进行分析
numeric_cols = [
    'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
    'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM'
]

# 确保只保留存在的列
available_cols = [col for col in numeric_cols if col in df.columns]
numeric_df = df[available_cols].copy()

# 转换为数值型（防止字符串干扰）
for col in numeric_df.columns:
    numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

# 删除全是 NaN 的列
numeric_df.dropna(axis=1, how='all', inplace=True)

# 3. 设置绘图风格
# sns.set(style="whitegrid")

# 4. 创建子图画布（每个变量画两个图：直方图 + 箱线图）
num_vars = len(numeric_df.columns)
fig, axes = plt.subplots(num_vars, 2, figsize=(10, 4 * num_vars))  # 每行两个图
axes = axes.flatten()  # 展平为一维数组以便循环使用

# 5. 绘制每个字段的直方图 + 箱线图
for i, col in enumerate(numeric_df.columns):
    # 直方图
    sns.histplot(numeric_df[col].dropna(), ax=axes[2*i], kde=True)
    axes[2*i].set_title(f'Histogram of {col}')
    
    # 箱线图
    sns.boxplot(x=numeric_df[col], ax=axes[2*i + 1])
    axes[2*i + 1].set_title(f'Boxplot of {col}')

# 自动调整布局
plt.tight_layout()
plt.savefig('variable_distributions.png', dpi=150, bbox_inches='tight')
plt.show()