import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from load_data import load_data_campaign1, load_data_campaign2, load_data_campaign3

# Odabir kampanje
kampanja = "campaign2" 

# Učitavanje podataka na temelju kampanje
if kampanja == "campaign1":
    train_data, test_data = load_data_campaign1()
elif kampanja == "campaign2":
    train_data, test_data = load_data_campaign2()
elif kampanja == "campaign3":
    train_data, test_data = load_data_campaign3()
else:
    print("Greška: Nevažeća kampanja!")
    exit()

# Provjera postoji li podatak
if train_data is None or test_data is None:
    print("Greška pri učitavanju podataka!")
    exit()

# Odabir značajki 
X_train = train_data[['speed', 'acceleration', 'wind', 'traffic_factor']]  
y_train = train_data['total_energy_consumed_wh'] 

X_test = test_data[['speed', 'acceleration', 'wind', 'traffic_factor']]  
y_test = test_data['total_energy_consumed_wh'] 

# Treniranje Random Forest modela
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predikcija za Random Forest
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

# Evaluacija za Random Forest
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

# Evaluacija za Gradient Boosting
gb_train_mse = mean_squared_error(y_train, gb_train_pred)
gb_train_r2 = r2_score(y_train, gb_train_pred)
gb_test_mse = mean_squared_error(y_test, gb_test_pred)
gb_test_r2 = r2_score(y_test, gb_test_pred)

# Ispis rezultata za oba modela
print("Random Forest Regressor:")
print(f"Train Mean Squared Error: {rf_train_mse:.2f}")
print(f"Train R-squared: {rf_train_r2:.2f}")
print(f"Test Mean Squared Error: {rf_test_mse:.2f}")
print(f"Test R-squared: {rf_test_r2:.2f}")

print("\nGradient Boosting Regressor:")
print(f"Train Mean Squared Error: {gb_train_mse:.2f}")
print(f"Train R-squared: {gb_train_r2:.2f}")
print(f"Test Mean Squared Error: {gb_test_mse:.2f}")
print(f"Test R-squared: {gb_test_r2:.2f}")