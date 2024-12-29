###########################################################################
#       DEVST: Battery Range Prediction Models                          #
#                     Zoe Bakalov                                        #
###########################################################################

Title
=====

Battery Range Prediction Models

Description
===========

This repository contains machine learning models developed to predict the battery range of electric vehicles based on the DEVST dataset. The models utilize key features such as speed, acceleration, wind conditions, and traffic factors to estimate remaining range and total energy consumption. Three different campaigns of data were used to train and evaluate these models, focusing on diverse aspects of electric vehicle performance.

Key Files
=========

- **load_data.py**: Provides functions to load training and test datasets for different campaigns.
- **model1.py**: Implements a Random Forest Regressor to predict remaining range, while handling highly correlated features.
- **model2.py**: Compares the performance of Random Forest and Gradient Boosting Regressors for predicting total energy consumption.
- **model3.py**: A Random Forest Classifier that categorizes completed distance into three distinct ranges (<50 km, 50-150 km, >150 km).


Models and Results
==================

### Model 1: Random Forest Regressor for Remaining Range

- **Objective**: Predict the remaining range (km) while addressing feature correlations.
- **Key Results**:
  - Training MSE: ~[10398080.0481]
  - Training R²: ~[0.7385]
  - Test MSE: ~[13643818.7389]
  - Test R²: ~[-0.1419]

### Model 2: Random Forest vs. Gradient Boosting Regressors

- **Objective**: Predict total energy consumption (Wh) and compare model performance.
- **Key Results (Random Forest)**:
  - Training MSE: ~[12467774.06]
  - Training R²: ~[0.91]
  - Test MSE: ~[36756788.09]
  - Test R²: ~[0.72]

- **Key Results (Gradient Boosting)**:
  - Training MSE: ~[105894948.00]
  - Training R²: ~[0.20]
  - Test MSE: ~[105960638.54]
  - Test R²: ~[0.20]


### Model 3: Random Forest Classifier for Range Categories

- **Objective**: Classify completed distances into categories (<50 km, 50-150 km, >150 km).
- **Key Results**:
  - Training Accuracy: ~[0.9985]
  - Test Accuracy: ~[0.9343]

How to Use
==========

### Step 1: Load Data

Use the `load_data.py` script to load the appropriate campaign dataset:

```python
from load_data import load_data_campaign1
train_data, test_data = load_data_campaign1()
```

### Step 2: Run Models

Run any of the model scripts to train and evaluate the respective models:

```bash
python model1.py
python model2.py
python model3.py
```

### Step 3: Visualize Results

Plots and metrics will be generated for model evaluation, such as feature importance and actual vs. predicted values.

License
=======
All datasets and code in this repository are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

Citation
========
If you find this useful for you project or research, consider citing it.

