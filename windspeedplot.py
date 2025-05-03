import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if  __name__ == '__main__':
    features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'] 

    df = pd.read_csv('processed_data.csv', 
                    usecols=['WSPM'] + features,
                    na_values=['NA', 'NaN', 'nan'])



    df['WSPM_bins'] = pd.cut(df['WSPM'], bins=np.arange(0, df['WSPM'].max() + 0.5, 0.5))

    fig = plt.figure(figsize=(20, 10))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for i, feature in enumerate(features, start=1):
        # 计算每个风速区间内各特征的平均值
        mean_vals = df.groupby('WSPM_bins', observed=True)[feature].mean()
        
        plt.subplot(2, 3, i) # 2行3列网格中的第i个子图
        mean_vals.plot(kind='bar', ax=plt.gca())
        plt.title(f'WSPM vs Avg {feature}')
        plt.xlabel('WSPM Range')
        plt.ylabel(f'Avg {feature}')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('windspeed_airquality.png')