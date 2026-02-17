import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# read the two datasets and combined them on date

weather = pd.read_csv("weather_clean_2023.csv")
bike = pd.read_csv("citibike_summary_2023.csv")

# let the format be same

weather["date"] = pd.to_datetime(weather["DATE"])  
bike["date"] = pd.to_datetime("2023-" + bike["date"], format="%Y-%m-%d")  # "2023-01-01"
weather = weather.drop(columns=["DATE"])
merged = pd.merge(weather, bike, on="date", how="inner")

# move column

cols = ["date"] + [c for c in merged.columns if c != "date"]
merged = merged[cols]

merged.to_csv("merged_weather_citibike_2023.csv", index=False)

# EDA

# statistics
print("\n==== Summary Statistics (numeric) ====")
print(merged.describe())


# Distribution of trips
plt.figure()
plt.hist(merged["trips"], bins=30)
plt.title("Distribution of Daily Trips")
plt.xlabel("Trips")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# trips vs date
plt.figure()
plt.plot(merged["date"], merged["trips"])
plt.title("Daily Trips Trend (2023)")
plt.xlabel("Date")
plt.ylabel("Trips")
plt.tight_layout()
plt.show()

# tempurature vs trips

plt.figure()
plt.scatter(merged["TEMP_AVG"], merged["trips"], s=10)
plt.title("TEMP_AVG vs Trips")
plt.xlabel("Average Temperature (TEMP_AVG)")
plt.ylabel("Trips")
plt.tight_layout()
plt.show()

# Data Preprocessing & Feature Engineering

# To enhance the predictive usefulness of the dataset, 
# additional temporal features were derived from the original date variable. 
# Specifically, the month and day of the week were extracted using datetime operations.
merged["month"] = merged["date"].dt.month
merged["dayofweek"] = merged["date"].dt.dayofweek  # 0=Mon ... 6=Sun


# This engineered feature integrates temperature and 
# precipitation into a single interpretable metric 
# representing how favorable daily weather conditions are for cycling activity.
merged["weather_comfort"] = merged["TEMP_AVG"] - 0.1 * merged["PRCP"]

# Standardization of Numerical Variables

# All numerical features were standardized using z-score normalization implemented via StandardScaler. 
# This transformation rescales each variable so that:
# The mean equals 0
# The standard deviation equals 1
numeric_cols = merged.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()
merged_scaled = merged.copy()
merged_scaled[numeric_cols] = scaler.fit_transform(merged_scaled[numeric_cols])

# Save
merged_scaled.to_csv("preprocessed_weather_citibike_2023.csv", index=False)