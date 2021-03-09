from flask import Flask, jsonify, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("boston.joblib")
fake_model = joblib.load("fake.joblib")


@app.route("/", methods=["POST"])
def index():
    y = model.predict(request.json)
    return jsonify(y.tolist())


@app.route("/fake", methods=["GET", "POST"])
def fake():
    print(request.method)
    if request.method == "POST":
        x = float(request.form["x"])
        y = fake_model.predict([[x]])
        print(y)
    return render_template("fake.html", y=str(y) or "")
