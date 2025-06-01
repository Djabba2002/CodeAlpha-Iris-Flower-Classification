from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

target_names = ["setosa", "versicolor", "virginica"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            features = [
                float(request.form["sepal_length"]),
                float(request.form["sepal_width"]),
                float(request.form["petal_length"]),
                float(request.form["petal_width"])
            ]
            data = np.array([features])
            pred_index = model.predict(data)[0]
            prediction = target_names[pred_index]
        except ValueError:
            prediction = "Entrée invalide. Veuillez utiliser des chiffres valides."
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)