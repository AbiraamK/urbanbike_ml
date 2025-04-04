import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data(file_path):
    #Loads and preprocesses the dataset
    bike_data = pd.read_csv(file_path)
    bike_data.drop(columns=["instant", "dteday", "casual", "registered"], inplace=True)
    X = bike_data.drop(columns=["cnt"])
    y = bike_data["cnt"]
    return X, y

def preprocess_data(X):
    #Defines preprocessing steps for numerical and categorical data
    num_features = ["temp", "atemp", "hum", "windspeed"]
    cat_features = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    return preprocessor

def train_models(X_train, X_test, y_train, y_test, preprocessor):
    #Trains multiple models and evaluates performance
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "R2": r2}
        print(f"{name} -> MSE: {mse:.2f}, R2: {r2:.2f}")
    return results

def hyperparameter_tuning(X_train, y_train, preprocessor, models):
    #Performs hyperparameter tuning on the best models (Random Forest and Gradient Boosting)
    param_grids = {
        "Random Forest": {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 10],
            "model__min_samples_split": [2, 5]
        },
        "Gradient Boosting": {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5]
        }
    }
    
    best_models = {}
    for name, param_grid in param_grids.items():
        print(f"Tuning {name}...")
        grid_search = GridSearchCV(Pipeline([
            ('preprocessor', preprocessor),
            ('model', models[name])
        ]), param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best Parameters for {name}: {grid_search.best_params_}")
    
    return best_models

def main():
    #Main function to run the training and evaluation
    file_path = "hour.csv"  
    X, y = load_data(file_path)
    preprocessor = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    
    results = train_models(X_train, X_test, y_train, y_test, preprocessor)
    best_models = hyperparameter_tuning(X_train, y_train, preprocessor, models)
    
    for name, model in best_models.items():
        y_pred_best = model.predict(X_test)
        print(f"Final {name} -> MSE: {mean_squared_error(y_test, y_pred_best):.2f}, R2: {r2_score(y_test, y_pred_best):.2f}")
        joblib.dump(model, f"best_{name.replace(' ', '_').lower()}_model.pkl")
        print(f"Best {name} model saved.")

if __name__ == "__main__":
    main()
