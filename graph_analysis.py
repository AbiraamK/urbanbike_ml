import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = "hour.csv"
bike_data = pd.read_csv(file_path)
X = bike_data.drop(columns=["cnt"])
y = bike_data["cnt"]

# Load the trained models
rf_model = joblib.load("best_random_forest_model.pkl")
gb_model = joblib.load("best_gradient_boosting_model.pkl")

# Get predictions
rf_pred = rf_model.predict(X)
gb_pred = gb_model.predict(X)

# Evaluate models
rf_mse = mean_squared_error(y, rf_pred)
rf_r2 = r2_score(y, rf_pred)
gb_mse = mean_squared_error(y, gb_pred)
gb_r2 = r2_score(y, gb_pred)

# Model Performance Bar Chart 
model_performance = {
    "Random Forest": {"MSE": rf_mse, "R²": rf_r2},
    "Gradient Boosting": {"MSE": gb_mse, "R²": gb_r2},
}

df_perf = pd.DataFrame(model_performance).T

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

df_perf["MSE"].plot(kind="bar", ax=ax1, color="skyblue", width=0.4, position=1, label="MSE")
df_perf["R²"].plot(kind="bar", ax=ax2, color="darkblue", width=0.4, position=0, label="R² Score")

ax1.set_xlabel("Models", fontsize=12)
ax1.set_ylabel("MSE (Lower is Better)", fontsize=12, color="blue")
ax2.set_ylabel("R² Score (Higher is Better)", fontsize=12, color="darkblue")
ax1.set_title("Comparison of Model Performance", fontsize=14, fontweight="bold")
ax1.grid(axis="y", linestyle="--", alpha=0.6)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.xticks(rotation=0, fontsize=10)
plt.show()

#  Scatter Plot: Actual vs Predicted 
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Random Forest
sns.scatterplot(x=y, y=rf_pred, alpha=0.5, ax=axes[0], color='blue')
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "--", color="red")
axes[0].set_title("Random Forest: Actual vs. Predicted")
axes[0].set_xlabel("Actual Bike Rentals")
axes[0].set_ylabel("Predicted Bike Rentals")
axes[0].grid(True, linestyle="--", alpha=0.6)

# Gradient Boosting
sns.scatterplot(x=y, y=gb_pred, alpha=0.5, ax=axes[1], color='orange')
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], "--", color="red")
axes[1].set_title("Gradient Boosting: Actual vs. Predicted")
axes[1].set_xlabel("Actual Bike Rentals")
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

#  Line Graph: Bike Rentals by Hour 
plt.figure(figsize=(10, 6))
sns.lineplot(data=bike_data.groupby("hr")["cnt"].mean(), marker="o", color="darkorange")
plt.xlabel("Hour of the Day", fontsize=12)
plt.ylabel("Average Bike Rentals", fontsize=12)
plt.title("Bike Rentals by Hour", fontsize=14, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(range(0, 24))
plt.show()