# 🚲 Urban Bike ML – Bike Rental Demand Prediction

This project analyzes and predicts hourly bike rental demand using machine learning models like Random Forest and Gradient Boosting. The dataset includes weather, temporal, and seasonal information from a bike-sharing system.

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `bike_project_main.py` | **Main script** to train models and generate `.pkl` files |
| `graph_analysis.py` | Generates performance graphs after training |
| `graph_scatter_plots.py` | Produces actual vs. predicted scatter plots only |
| `hour.csv` | Dataset used to train models |
| `*.pkl` | Saved model files (generated after running the main script) |
| `*.png` | Visualizations exported from the analysis |

---

## 🧠 Models Used

- Random Forest Regressor
- Gradient Boosting Regressor

The models are trained and tuned using grid search, and saved as `.pkl` files using `joblib`.

---

## 🛠 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/AbiraamK/urban_bike_ml.git
cd urban_bike_ml
```

### 2. Install required libraries
```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib
```

### 3. Run the main training script
📌 This generates the model files: `best_random_forest_model.pkl` and `best_gradient_boosting_model.pkl`
```bash
python bike_project_main.py
```

### 4. Run the visualizations
**For full graphs:**
```bash
python graph_analysis.py
```

**For scatter plots only:**
```bash
python graph_scatter_plots.py
```

---

## 📊 Sample Output Visualizations
<p float="left">
  <img src="Model Performance.png" width="300"/>
  <img src="Actual vs. Predicted Graph.png" width="300"/>
  <img src="Bike Rentals.png" width="300"/>
</p>

---


## 👨‍💻 Author
Abiraam K.  
[GitHub Profile](https://github.com/AbiraamK)

---

## 📄 License
This project is for educational purposes. Attribution appreciated.
