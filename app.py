from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("rf_model_joblib.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None

    if request.method == "POST":
        voltage = float(request.form["voltage"])
        temperature = float(request.form["temperature"])
        fan_speed = int(request.form["fan_speed"])
        battery = request.form["battery"]
        power_rail = request.form["power_rail"]
        beep = request.form["beep"]

        data = {
            "Voltage": voltage,
            "Temperature": temperature,
            "Fan_Speed": fan_speed,
            "Battery_Health": battery,
            "Power_Rail": power_rail,
            "Beep_Code": beep
        }

        df = pd.DataFrame([data])
        df_encoded = pd.get_dummies(df)

        for col in model.feature_names_in_:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[model.feature_names_in_]

        pred = model.predict(df_encoded)[0]
        prob = max(model.predict_proba(df_encoded)[0])

        # Certainty Factor
        cf = {
            "Overheating": (1 if temperature > 75 else 0) * 0.7 + (1 if fan_speed == 0 else 0) * 0.3,
            "Power Supply": (1 if voltage < 10 else 0) * 0.8 + (1 if power_rail == "Failed" else 0) * 0.2,
            "Battery Issue": (1 if battery in ["Poor", "Bad"] else 0) * 0.9,
            "Motherboard": (1 if beep in ["Long", "Continuous"] else 0) * 0.9
        }

        alpha = 0.5
        final = {}
        for k in cf:
            final[k] = alpha * (prob if k == pred else 0) + (1 - alpha) * cf[k]

        hasil = {
            "pred": pred,
            "prob": round(prob, 2),
            "cf": cf,
            "final": final,
            "diagnosis": max(final, key=final.get)
        }

    return render_template("index.html", hasil=hasil)

if __name__ == "__main__":
    app.run(debug=True)
