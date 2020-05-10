from flask import Flask, jsonify, request
from flask_cors import CORS
from joblib import dump, load
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

CLASSIFIER_FILE_NAME = "classifier.joblib"

# Initialize model
dump(LogisticRegression(multi_class="ovr", random_state=0, solver="lbfgs"), CLASSIFIER_FILE_NAME)

@app.route("/train", methods=["POST"])
def train():
    # Get data from post method
    data = request.get_json()

    # Load classifier
    classifier = load(CLASSIFIER_FILE_NAME)

    # fit the model
    x = np.array(data["x"]).reshape(-1, 1)
    y = np.array(data["y"])
    classifier.fit(x, y)

    # Re-store the model
    dump(classifier, CLASSIFIER_FILE_NAME)

    # Return an empty response
    return jsonify({}), 200

@app.route("/classify/<string:distance>", methods=["GET"])
def classify(distance):
    try:
        distance = float(distance)
    except Exception:
        # Some garbage was passed to the distance parameter
        return jsonify({}), 500

    try:
        # Load classifier
        classifier = load(CLASSIFIER_FILE_NAME)

        # Make prediction
        _, accept_chance = classifier.predict_proba(np.array([distance]).reshape(-1, 1))[0]
        return jsonify({"result": accept_chance}), 200
    except Exception:
        # The model wasn't fitted first somehow
        return jsonify({}), 500

if __name__ == "__main__":
    app.run(debug=False)
