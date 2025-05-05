# distribution.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    df = pd.read_csv('data.csv', na_values=['NA', 'NaN', 'nan'])
    numeric_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
                    'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    available_cols = [col for col in numeric_cols if col in df.columns]
    numeric_df = df[available_cols].copy()

    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

    num_vars = len(numeric_df.columns)

    fig_hist, axes_hist = plt.subplots(3, 2, figsize=(20, 15))
    axes_hist = axes_hist.flatten()

    for i, col in enumerate(numeric_df.columns):
        sns.histplot(numeric_df[col].dropna(), ax=axes_hist[i], kde=True)
        axes_hist[i].set_title(f'Histogram of {col}')
        axes_hist[i].set_xlabel(col)
        axes_hist[i].set_ylabel('Frequency')

    # 隐藏多余的空子图（如果字段数 < 12）
    for j in range(i + 1, len(axes_hist)):
        fig_hist.delaxes(axes_hist[j])

    plt.tight_layout()
    plt.savefig('histograms_all.png', dpi=150, bbox_inches='tight')

    fig_box, axes_box = plt.subplots(3, 4, figsize=(20, 15))
    axes_box = axes_box.flatten()

    for i, col in enumerate(numeric_df.columns):
        sns.boxplot(x=numeric_df[col], ax=axes_box[i], whis=3.0)
        axes_box[i].set_title(f'Boxplot of {col}')
        axes_box[i].set_xlabel(col)

    for j in range(i + 1, len(axes_box)):
        fig_box.delaxes(axes_box[j])

    plt.tight_layout()
    plt.savefig('boxplots_all.png', dpi=150, bbox_inches='tight')