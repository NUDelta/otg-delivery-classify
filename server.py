from flask import Flask, jsonify, request
# from flask_cors import CORS
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
# cors = CORS(app, resources={"/", {"Access-Control-Allow-Origin": "*"}})

# Sample Data
X = np.array([15, 25, 36, 26, 80, 14, 15, 43, 51, 36, 38, 22, 60, 65, 73, 45, 15, 77, 94, 83, 51, 55, 65, 55]).reshape(-1, 1)
y = np.array([1] * 12 + [0] * 12)

DEFAULT_PROBABILITY = 0.3

# https://scikit-learn.org/stable/modules/model_persistence.html -> saving model
classifier = LogisticRegression(random_state=0, multi_class="ovr")
# classifier.fit(X, y)

def response(val=DEFAULT_PROBABILITY):
    return jsonify({"result": val})

@app.route("/train", methods=["POST"])
def train():
    # get X and y data from post method
    data = request.form
    print(data.x, data.y)

    # fit the model
    classifier.fit(X, y)

@app.route("/classify/<string:distance>", methods=["GET"])
def classify(distance):
    try:
        distance = float(distance)
    except Exception:
        # Some garbage was passed in
        return response()

    try:
        _, accept_chance = classifier.predict_proba(np.array([15]).reshape(-1, 1))[0]
        return response(accept_chance)
    except Exception:
        # The model wasn't fitted first somehow
        return response()

if __name__ == "__main__":
    app.run(debug=True)
