import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from load_data import load_data_campaign1, load_data_campaign2, load_data_campaign3
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder

kampanja = "campaign2"

# Učitavanje podataka 
kampanja = "campaign1"
if kampanja == "campaign1":
    train_data, test_data = load_data_campaign1()
elif kampanja == "campaign2":
    train_data, test_data = load_data_campaign2()
else:
    train_data, test_data = load_data_campaign3()


# Prebacivanje svih tekstualnih podataka u numeričke (Label Encoding za 'vehicle_id')
label_encoder = LabelEncoder()

# Fitiranje LabelEncoder-a samo na trening podatke za 'vehicle_id'
train_data['vehicle_id'] = label_encoder.fit_transform(train_data['vehicle_id'])

# Provjera i transformacija test podataka za 'vehicle_id'
train_vehicle_ids = set(train_data['vehicle_id'])
test_vehicle_ids = set(test_data['vehicle_id'])

# Dodavanje nepoznatih vrijednosti kao posebnu kategoriju za 'vehicle_id'
label_encoder.classes_ = np.append(label_encoder.classes_, 'unknown')

# Transformiraj testne podatke za 'vehicle_id'
test_data['vehicle_id'] = label_encoder.transform(test_data['vehicle_id'].apply(lambda x: x if x in train_vehicle_ids else 'unknown'))

# Odabir značajki koje ćemo koristiti za treniranje
X_train = train_data[['speed', 'acceleration', 'road_slope', 'auxiliaries', 'traffic_factor', 'wind']]  
y_train = train_data['total_energy_consumed_wh'] 

X_test = test_data[['speed', 'acceleration', 'road_slope', 'auxiliaries', 'traffic_factor', 'wind']]  
y_test = test_data['total_energy_consumed_wh']

# Treniranje Random Forest modela
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predikcija za Random Forest
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

# Rezultati za Random Forest
rf_train_mse = mean_squared_error(y_train, rf_train_pred)
rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_test_mse = mean_squared_error(y_test, rf_test_pred)
rf_test_r2 = r2_score(y_test, rf_test_pred)

# Treniranje Gradient Boosting modela
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Predikcija za Gradient Boosting
gb_train_pred = gb_model.predict(X_train)
gb_test_pred = gb_model.predict(X_test)

# Rezultati za Gradient Boosting
gb_train_mse = mean_squared_error(y_train, gb_train_pred)
gb_train_r2 = r2_score(y_train, gb_train_pred)
gb_test_mse = mean_squared_error(y_test, gb_test_pred)
gb_test_r2 = r2_score(y_test, gb_test_pred)

# Ispis rezultata za oba modela
print("\nRandom Forest Regressor:")
print(f"Train Mean Squared Error: {rf_train_mse:.2f}")
print(f"Train R-squared: {rf_train_r2:.2f}")
print(f"Test Mean Squared Error: {rf_test_mse:.2f}")
print(f"Test R-squared: {rf_test_r2:.2f}")

print("\nGradient Boosting Regressor:")
print(f"Train Mean Squared Error: {gb_train_mse:.2f}")
print(f"Train R-squared: {gb_train_r2:.2f}")
print(f"Test Mean Squared Error: {gb_test_mse:.2f}")
print(f"Test R-squared: {gb_test_r2:.2f}")

# Histogram predikcija vs stvarnih vrijednosti
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(rf_test_pred, kde=True, color='blue', label="Predikcija (RF)", stat="density")
sns.histplot(y_test, kde=True, color='red', label="Stvarne vrijednosti", stat="density")
plt.title('Distribucija: Predikcija vs Stvarne vrijednosti (Random Forest)')
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(gb_test_pred, kde=True, color='blue', label="Predikcija (GB)", stat="density")
sns.histplot(y_test, kde=True, color='red', label="Stvarne vrijednosti", stat="density")
plt.title('Distribucija: Predikcija vs Stvarne vrijednosti (Gradient Boosting)')
plt.legend()

plt.tight_layout()
plt.show()
