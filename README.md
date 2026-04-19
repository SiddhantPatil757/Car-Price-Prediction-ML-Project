#  Car Price Prediction System

An end-to-end Machine Learning project that predicts the selling price of used cars based on features like mileage, age, fuel type, and transmission.

---

##  Highlights
- Built using **Python, Scikit-learn, Flask**
- Achieved **R² Score ≈ 0.96**
- Developed both **REST API + Web Interface**
- Real-time prediction using trained ML model
- Clean modular project structure

---
##  Demo

![Car Price Prediction App](screenshot.png)
---

##  Problem Statement
To build a machine learning model that predicts the price of used cars based on historical data and helps users make informed buying and selling decisions.

---

##  Tech Stack

| Category   |               Tools         |
|------------|-----------------------------|
| Language   | Python                      |
| Libraries  | Pandas, NumPy, Scikit-learn |
| Model      | Random Forest Regressor     |
| Backend    | Flask                       |
| Frontend   | HTML                        |
| Deployment | Local (Flask server)        |

---

##  Project Structure
car-price-prediction/
│
├── data/
│ └── raw/
│ └── car_data.csv
│
├── models/
│ └── model.pkl
│
├── notebooks/
│ └── eda.ipynb
│
├── src/
│ ├── init.py
│ ├── data_preprocessing.py
│ ├── train.py
│ └── predict.py
│
├── templates/
│ └── index.html
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

---

##  Features Used

- **Present Price** → Original showroom price (in lakhs)
- **Kms Driven** → Distance travelled
- **Fuel Type** → Petrol / Diesel / CNG
- **Seller Type** → Dealer / Individual
- **Transmission** → Manual / Automatic
- **Owner** → Number of previous owners
- **Car Age** → Years since purchase

---

##  Machine Learning Workflow

1. Data Collection & Cleaning  
2. Feature Engineering (Car Age creation)  
3. Encoding categorical variables  
4. Train-Test Split  
5. Model Training using Random Forest  
6. Model Evaluation  
7. Model Saving using Joblib  
8. Deployment using Flask  

---

##  Model Performance

| Metric   | Value |
|----------|-------|
| R² Score | ~0.96 |
| MAE      | Low   |
| RMSE     | Low   |

---

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction