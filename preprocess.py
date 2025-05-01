from matplotlib import pyplot as plt
import pandas as pd


df = pd.read_csv('data.csv', na_values=['NA'])  
df_cleaned = df.dropna()

df_cleaned['datetime'] = pd.to_datetime(
    df_cleaned[['year', 'month', 'day', 'hour']]
)

plt.figure(figsize=(8, 6))
plt.suptitle('Environmental Indicators Time Series', y=1.02)

metrics = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES']

# 创建2x4的子图画布
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 4, i)
    plt.plot(df_cleaned['datetime'], df_cleaned[metric], 
            label=metric, linewidth=1)
    plt.title(f'{metric} Trend')
    plt.xlabel('Date')
    plt.ylabel(metric)
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

hourly_mean = df_cleaned.groupby('hour').mean(numeric_only=True)
plt.figure(figsize=(8, 6))
plt.suptitle('Hourly Variation Patterns', y=1.02)

metrics = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES']

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 4, i)
    
    plt.bar(hourly_mean.index, hourly_mean[metric], 
            color='steelblue', alpha=0.8)
    
    plt.title(f'{metric} Hourly Variation')
    plt.xlabel('Hour of Day')
    plt.ylabel(metric)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(range(0,24))

plt.tight_layout()
plt.show()