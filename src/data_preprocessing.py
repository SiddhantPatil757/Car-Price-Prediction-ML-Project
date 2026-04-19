import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.copy()

    # create car age
    df["Car_Age"] = 2026 - df["Year"]
    df.drop(["Year","Car_Name"],axis=1,inplace=True)

    # Encode categorical 
    le = LabelEncoder()
    for col in ["Fuel_Type","Seller_Type","Transmission"]:
        df[col] = le.fit_transform(df[col])

    return df