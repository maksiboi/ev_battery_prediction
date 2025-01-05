# app/routes.py
import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend for Flask

import matplotlib.pyplot as plt
from flask import Blueprint, jsonify, Response
import pickle
import os
import io
from app.load_data import load_data
from sklearn.preprocessing import LabelEncoder

bp = Blueprint("main", __name__)

# Get the absolute path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../data/model_data/model1.pkl")
# Resolve to an absolute path
MODEL_PATH = os.path.abspath(MODEL_PATH)
CAMPAIGN = "campaign1"

@bp.route("/", methods=["GET"])
def predict_remaining_range():
    # Loading a pre-trained machine learning model that has been saved using pickle.
    # This model is stored in a file and is loaded as follows:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load test data
    _, test_data = load_data(CAMPAIGN)

    # Select first 5 rows for prediction (DEMO PURPOSES ONLY)
    test_data = test_data.head(5)

    if test_data is None or test_data.empty:
        return jsonify({"error": "Test data is missing or empty"}), 400

    # Exclude target column and drop NaN values
    X_test = test_data.drop(columns=["remaining_range"], errors="ignore").dropna()

    # Encode categorical features if present
    encoders = {"vehicle_id": LabelEncoder(), "trip_plan": LabelEncoder()}

    for col, encoder in encoders.items():
        if col in X_test.columns:
            X_test[col] = encoder.fit_transform(X_test[col])

    # After loading the model and the test data, the backend processes the test data by encoding any categorical features and then uses the model to make predictions
    predictions = model.predict(X_test)

    return jsonify({"predictions": predictions.tolist()})


@bp.route("/graphs", methods=["GET"])
def plot_predictions():
    """Generates and returns a scatter plot of actual vs predicted values."""
    # Load the trained model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load test data
    _, test_data = load_data(CAMPAIGN)
    test_data = test_data.head(5)

    if test_data is None or test_data.empty:
        return jsonify({"error": "Test data is missing or empty"}), 400

    # Prepare data for plotting
    X_test = test_data.drop(columns=["remaining_range"], errors="ignore").dropna()
    y_test = test_data.loc[X_test.index, "remaining_range"]

    # Encode categorical columns
    encoders = {"vehicle_id": LabelEncoder(), "trip_plan": LabelEncoder()}
    for col, encoder in encoders.items():
        if col in X_test.columns:
            X_test[col] = encoder.fit_transform(X_test[col])

    # Make predictions
    predictions = model.predict(X_test)

    # Create scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, predictions, color="blue", label="Predictions")
    plt.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        color="red",
        linestyle="dashed",
        label="Ideal Fit",
    )
    plt.xlabel("Actual Remaining Range")
    plt.ylabel("Predicted Remaining Range")
    plt.title("Actual vs. Predicted Remaining Range")
    plt.legend()

    # Save plot to a BytesIO object and return as a response
    img = io.BytesIO()
    plt.savefig(img, format="png")
    plt.close()  # Close the figure to free memory
    img.seek(0)

    return Response(img.getvalue(), mimetype="image/png")


def configure_routes(app):
    app.register_blueprint(bp)
