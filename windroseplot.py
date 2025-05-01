import pandas as pd
import matplotlib.pyplot as plt
import windrose

features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'] 

df = pd.read_csv('processed_data.csv', 
                usecols=['wd'] + features,
                na_values=['NA', 'NaN', 'nan'])

# 绘制图像
fig = plt.figure(figsize=(20, 10))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

for i, feature in enumerate(features, 1):
    ax = fig.add_subplot(2, 4, i, projection="windrose")
    
    ax.bar(df['wd'], 
           df[feature],
           bins=8,             # 按特征值分8个区间
           cmap=plt.cm.plasma,
           edgecolor='gray')
    
    ax.set_title(f"Wind with {feature}", pad=20)
    ax.set_legend(title=feature + ' Values', bbox_to_anchor=(1.1, 0.5))

plt.suptitle("Multi-Feature Wind Rose Analysis", y=0.95, fontsize=14)
plt.tight_layout()
plt.savefig('multi_feature_wind_rose.png')