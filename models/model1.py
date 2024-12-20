import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from load_data import load_data_campaign1, load_data_campaign2, load_data_campaign3

kampanja = "campaign1"

if kampanja == "campaign1":
    train_data, test_data = load_data_campaign1()
elif kampanja == "campaign2":
    train_data, test_data = load_data_campaign2()
else:
    train_data, test_data = load_data_campaign3()

X_train = train_data[['state_of_charge', 'speed', 'acceleration', 'completed_distance', 'wind', 'traffic_factor']]
y_train = train_data['remaining_range']
X_test = test_data[['state_of_charge', 'speed', 'acceleration', 'completed_distance', 'wind', 'traffic_factor']]
y_test = test_data['remaining_range']

# Inicijalizacija modela
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Predikcija
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrike
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Ispis rezultata
print(f"Trening MSE: {train_mse:.4f}")
print(f"Trening R²: {train_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Vizualizacija: stvarne vs predikcije
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue', label='Predikcije')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Stvarne vrijednosti')
plt.ylabel('Predikcije')
plt.title(f'Stvarne vs Predikcije ({kampanja})')
plt.legend()
plt.show()