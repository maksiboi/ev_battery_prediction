# Battery Range Prediction API

## Overview

This repository contains a Flask-based API that allows for predicting the remaining battery range of electric vehicles (EVs) using a pre-trained machine learning model. The application loads a machine learning model and test data, makes predictions, and returns the results as a JSON response or as a graph comparing predicted and actual values.

### Project Structure
    |____ run.py
    |____ app
    |      |____ __init__.py
    |      |____ load_data.py
    |      |____ routes.py
    |____ start_backend.sh

- **`run.py`**: The entry point to run the Flask application.
- **`app/`**: Contains the core components of the application.
  - **`__init__.py`**: Initializes the Flask application.
  - **`load_data.py`**: Handles loading of test data for prediction.
  - **`routes.py`**: Contains the API routes for predictions and graph generation.
- **`start_backend.sh`**: Shell script to start the Flask application.

## Start APP
```bash
cd backend
chmod +x start_backend.sh
./start_backend.sh
