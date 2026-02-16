'''
title: "Weather 2023"
author: "Yiyu Chen"
'''

import pandas as pd
weather = pd.read_csv(
    '/Users/caroline/Desktop/DATA SCIENCE/weather dataset.csv',
    dtype=str
)
weather['DATE'] = pd.to_datetime(weather['DATE'])
weather = weather[weather['DATE'].dt.year == 2023]
weather = weather[['DATE', 'TMAX', 'TMIN', 'PRCP', 'SNOW', 'AWND']]

for col in ['TMAX', 'TMIN', 'PRCP']:
    weather[col] = pd.to_numeric(weather[col], errors='coerce')

# Unit conversion (NOAA typically uses 10 times the temperature unit)
weather['TMAX'] = weather['TMAX'] / 10
weather['TMIN'] = weather['TMIN'] / 10
weather['PRCP'] = weather['PRCP'] / 10

# Handling missing values
weather = weather.dropna()

# Feature Engineering
weather['TEMP_AVG'] = (weather['TMAX'] + weather['TMIN']) / 2
weather['RAIN_FLAG'] = (weather['PRCP'] > 0).astype(int)
weather['HOT_DAY'] = (weather['TMAX'] > 30).astype(int)   # 30°C
weather['COLD_DAY'] = (weather['TMIN'] < 0).astype(int)
weather['WEEKDAY'] = weather['DATE'].dt.weekday
weather['IS_WEEKEND'] = weather['WEEKDAY'].isin([5,6]).astype(int)

# summary statistics
print(weather.describe())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

plt.figure(figsize=(12,6))

# Daily Temperature Trend
plt.plot(weather['DATE'], weather['TMAX'], label='Max Temp')
plt.plot(weather['DATE'], weather['TMIN'], label='Min Temp')
plt.plot(weather['DATE'], weather['TEMP_AVG'], label='Average Temp')
plt.title("Daily Temperature Trend in NYC (2023)")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()

# Precipitation Distribution
plt.figure(figsize=(8,5))
sns.histplot(weather['PRCP'], bins=30)
plt.title("Distribution of Daily Precipitation in NYC (2023)")
plt.xlabel("Precipitation (mm)")
plt.ylabel("Frequency")
plt.show()

# Wind Speed Distribution
plt.figure(figsize=(10,5))
sns.histplot(weather['AWND'], bins=20)
plt.title("Distribution of Average Wind Speed in NYC (2023)")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Frequency")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()