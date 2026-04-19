import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.data_preprocessing import load_data, preprocess_data


def train():
    # 📁 Robust path handling
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(BASE_DIR, "data", "raw", "car_data.csv")

    # 📊 Load + preprocess data
    df = load_data(data_path)
    df = preprocess_data(df)

    # 🎯 Features & Target
    X = df.drop("Selling_Price", axis=1)
    print("Feature Order:", list(X.columns))
    y = df["Selling_Price"]

    # 🔀 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🤖 Model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    # 🧠 Train
    model.fit(X_train, y_train)

    # 🔮 Predict
    preds = model.predict(X_test)

    # 📈 Evaluation Metrics
    print("R2 Score:", r2_score(y_test, preds))
    print("MAE:", mean_absolute_error(y_test, preds))
    print("MSE:", mean_squared_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))

    # 💾 Save Model
    model_path = os.path.join(BASE_DIR, "models", "model.pkl")
    joblib.dump(model, model_path)

    print("✅ Model saved at:", model_path)


if __name__ == "__main__":
    train()