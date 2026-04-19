import joblib
import numpy as np

def predict(features):
    model = joblib.load("models/model.pkl")
    features = np.array(features).reshape(1,-1)
    prediction = model.predict(features)
    return prediction[0]


