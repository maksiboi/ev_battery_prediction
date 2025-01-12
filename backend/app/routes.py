from flask import Blueprint, jsonify, Response, request
from flask_cors import CORS
import pickle
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


bp = Blueprint("main", __name__)
CORS(bp)

RANGE_MODEL_PATH = "insert path to model here"
RANGE_MODEL_PATH = os.path.abspath(RANGE_MODEL_PATH)

CONSUMPTION_MODEL_PATH = "insert path to model here"
CONSUMPTION_MODEL_PATH = os.path.abspath(CONSUMPTION_MODEL_PATH)

BATTERY_MODEL_PATH = "insert path to model here"
BATTERY_MODEL_PATH = os.path.abspath(BATTERY_MODEL_PATH)

@bp.route("/range/remaining", methods=["POST"])
def predict_remaining_range():
    with open(RANGE_MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    data = request.get_json()
    required_fields = [
        "vehicle_id", "trip_plan", "simulation_step", "acceleration", "actual_battery_capacity_wh",
        "state_of_charge", "speed", "total_energy_consumed_wh",
        "total_energy_regenerated_wh", "completed_distance",
        "traffic_factor", "wind", "remaining_range",
    ]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400


    test_data = pd.DataFrame([data])

    X_test = test_data.drop(columns=["remaining_range"], errors="ignore").dropna()

    encoders = {"vehicle_id": LabelEncoder(), "trip_plan": LabelEncoder()}
    for col, encoder in encoders.items():
        if col in X_test.columns:
            X_test[col] = encoder.fit_transform(X_test[col])

    prediction = model.predict(X_test)

    return jsonify({"prediction": prediction[0]})

@bp.route("/energy/consumption", methods=["POST"])
def predict_energy_consumption():
    with open(CONSUMPTION_MODEL_PATH, "rb") as f:
        model = pickle.load(f)

        data = request.get_json()

        required_fields = [
             "speed", "acceleration", "road_slope",
            "auxiliaries", "traffic_factor", "wind", "total_energy_consumed_wh"
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        test_data = pd.DataFrame([data])

        X_test = test_data.drop(columns=["total_energy_consumed_wh"], errors="ignore").dropna()

        encoders = {"vehicle_id": LabelEncoder()}
        for col, encoder in encoders.items():
            if col in X_test.columns:
                X_test[col] = encoder.fit_transform(X_test[col])

        prediction = model.predict(X_test)

        return jsonify({"prediction": prediction[0]}), 200

@bp.route("/battery/range", methods=["POST"])
def classify_battery_range():
    with open(BATTERY_MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    data = request.get_json()

    required_fields = [
        "speed", "acceleration",
        "completed_distance", "alt", "road_slope", "wind",
        "traffic_factor", "occupancy", "auxiliaries", "remaining_range"
    ]

    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

    test_data = pd.DataFrame([data])

    X_test = test_data.drop(columns=["remaining_range"], errors="ignore").dropna()

    encoders = {"vehicle_id": LabelEncoder()}
    for col, encoder in encoders.items():
        if col in X_test.columns:
            X_test[col] = encoder.fit_transform(X_test[col])

    prediction = model.predict(X_test)[0]

    range_categories = {
        0: "Short range (< 50 km)",
        1: "Medium range (50-150 km)",
        2: "Long range (> 150 km)"
    }
    range_category = range_categories.get(prediction, "Unknown category")

    return jsonify({"prediction": range_category}), 200

def configure_routes(app):
    app.register_blueprint(bp)
