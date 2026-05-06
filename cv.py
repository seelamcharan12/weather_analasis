import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# STEP 1: Generate Dataset
# -----------------------------
np.random.seed(42)

n = 100

dates = pd.date_range(start="2024-01-01", periods=n)

data = {
    "Date": dates,
    "Temperature": np.random.randint(15, 40, n),   # °C
    "Humidity": np.random.randint(30, 90, n),      # %
    "Wind_Speed": np.random.randint(1, 20, n),     # km/h
    "Rainfall": np.round(np.random.uniform(0, 20, n), 2)  # mm
}

df = pd.DataFrame(data)

# Save dataset
df.to_csv("weather_data.csv", index=False)

print("Dataset Created Successfully!\n")
print(df.head())

# -----------------------------
# STEP 2: Load & Preprocess
# -----------------------------
df = pd.read_csv("weather_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Add Month column
df["Month"] = df["Date"].dt.month

# -----------------------------
# STEP 3: Data Analysis
# -----------------------------
print("\n--- Data Analysis ---")

print("Average Temperature:", df["Temperature"].mean())
print("Maximum Temperature:", df["Temperature"].max())
print("Total Rainfall:", df["Rainfall"].sum())

print("\nMonthly Average Temperature:")
print(df.groupby("Month")["Temperature"].mean())

# -----------------------------
# STEP 4: Visualization
# -----------------------------

# 1. Temperature Trend
plt.figure()
plt.plot(df["Date"], df["Temperature"])
plt.title("Temperature Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Humidity Distribution
plt.figure()
sns.histplot(df["Humidity"], bins=20)
plt.title("Humidity Distribution")
plt.xlabel("Humidity (%)")
plt.ylabel("Frequency")
plt.show()

# 3. Temperature vs Rainfall
plt.figure()
sns.scatterplot(x=df["Temperature"], y=df["Rainfall"])
plt.title("Temperature vs Rainfall")
plt.xlabel("Temperature (°C)")
plt.ylabel("Rainfall (mm)")
plt.show()

# 4. Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Matrix")
plt.show()

# -----------------------------
# END
# -----------------------------
print("\nProject Completed Successfully!")