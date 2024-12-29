import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from load_data import load_data_campaign1, load_data_campaign2, load_data_campaign3
from sklearn.preprocessing import LabelEncoder  # Dodajemo LabelEncoder

# Učitavanje podataka 
kampanja = "campaign1"
if kampanja == "campaign1":
    train_data, test_data = load_data_campaign1()
elif kampanja == "campaign2":
    train_data, test_data = load_data_campaign2()
else:
    train_data, test_data = load_data_campaign3()

# Inicijalizacija LabelEncoder-a za enkodiranje 'vehicle_id' i 'trip_plan' kolona
vehicle_id_encoder = LabelEncoder()
trip_plan_encoder = LabelEncoder()

# Enkodiranje 'vehicle_id' i 'trip_plan' u numeričke vrijednosti
train_data['vehicle_id'] = vehicle_id_encoder.fit_transform(train_data['vehicle_id'])
test_data['vehicle_id'] = vehicle_id_encoder.transform(test_data['vehicle_id'])

train_data['trip_plan'] = trip_plan_encoder.fit_transform(train_data['trip_plan'])
test_data['trip_plan'] = trip_plan_encoder.transform(test_data['trip_plan'])

# Priprema podataka za model
remaining_features = [col for col in train_data.columns if col != "remaining_range"]

X_train = train_data[remaining_features]
y_train = train_data["remaining_range"]
X_test = test_data[remaining_features]
y_test = test_data["remaining_range"]

# Uklanjanje NaN vrijednosti ako ih ima
X_train = X_train.dropna()
y_train = y_train[X_train.index] 
X_test = X_test.dropna()
y_test = y_test[X_test.index] 

# Model koristeći najbolje hiperparametre
model = RandomForestRegressor(
    n_estimators=50,           
    max_depth=30,              
    min_samples_split=5,       
    min_samples_leaf=1,        
    max_features='sqrt',       
    random_state=42
)

# Treniranje modela
model.fit(X_train, y_train)

# Predikcija
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Rezultati
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Rezultati
print("\nRandom Forest:")
print(f"Trening MSE: {train_mse:.4f}")
print(f"Trening R²: {train_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Vizualizacija stvarnih vs. predviđenih vrijednosti za test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Stvarne vs. Predviđene Vrijednosti")
plt.xlabel("Stvarne Vrijednosti")
plt.ylabel("Predviđene Vrijednosti")
plt.show()
