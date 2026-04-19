

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]

    return jsonify({
        "predicted_price": float(prediction)
    })

@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        features = [
            float(request.form["present_price"]),
            float(request.form["kms_driven"]),
            int(request.form["fuel_type"]),
            int(request.form["seller_type"]),
            int(request.form["transmission"]),
            int(request.form["owner"]),
            int(request.form["car_age"])
        ]

        print("Input Features:", features)

        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)[0]

        present_price = features[0]

        if prediction > present_price * 1.2:
            prediction = present_price * 0.9  

        return render_template(
            "index.html",
            prediction_text=f"Predicted Price: ₹ {round(prediction, 2)} Lakhs"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )
    
if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)