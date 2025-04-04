import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset and models
file_path = "hour.csv"
bike_data = pd.read_csv(file_path)
X = bike_data.drop(columns=["cnt"])
y = bike_data["cnt"]

# Load models
rf_model = joblib.load("best_random_forest_model.pkl")
gb_model = joblib.load("best_gradient_boosting_model.pkl")

# Get predictions
rf_pred = rf_model.predict(X)
gb_pred = gb_model.predict(X)

# Create side-by-side plots
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
