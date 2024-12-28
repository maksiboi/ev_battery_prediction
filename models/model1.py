import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from load_data import load_data_campaign1, load_data_campaign2, load_data_campaign3

# Učitavanje podataka
kampanja = "campaign1"
if kampanja == "campaign1":
    train_data, test_data = load_data_campaign1()
elif kampanja == "campaign2":
    train_data, test_data = load_data_campaign2()
else:
    train_data, test_data = load_data_campaign3()

# Uklanjanje nenumeričkih kolona
train_data = train_data.select_dtypes(include=[np.number])
test_data = test_data.select_dtypes(include=[np.number])

# Analiza korelacije
correlation_matrix = train_data.corr()

# Pronalaženje visoko koreliranih varijabli
high_corr_threshold = 0.85
correlated_features = set()

for i in range(correlation_matrix.shape[0]):
    for j in range(i + 1, correlation_matrix.shape[1]):
        if abs(correlation_matrix.iloc[i, j]) > high_corr_threshold:
            feature1 = correlation_matrix.index[i]
            feature2 = correlation_matrix.columns[j]
            correlated_features.add((feature1, feature2))

# Uklanjanje jedne od koreliranih varijabli
features_to_remove = set([pair[1] for pair in correlated_features])

# Priprema podataka za model
remaining_features = [col for col in train_data.columns if col not in features_to_remove and col != "remaining_range"]

X_train = train_data[remaining_features]
y_train = train_data["remaining_range"]
X_test = test_data[remaining_features]
y_test = test_data["remaining_range"]

# Uklanjanje NaN vrednosti ako ih ima
X_train = X_train.dropna()
y_train = y_train[X_train.index]  # Sinkronizacija sa X_train
X_test = X_test.dropna()
y_test = y_test[X_test.index]  # Sinkronizacija sa X_test

# Random Forest model
model = RandomForestRegressor(
    n_estimators=50, 
    max_depth=20, 
    min_samples_split=10, 
    min_samples_leaf=2, 
    max_features='sqrt', 
    bootstrap=True,
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
print("\nRezultati Random Forest modela nakon uklanjanja visoko korelisanih karakteristika:")
print(f"Trening MSE: {train_mse:.4f}")
print(f"Trening R²: {train_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Vizualizacija stvarnih vs. predviđenih vrednosti za test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Stvarne vs. Predviđene Vrednosti")
plt.xlabel("Stvarne Vrednosti")
plt.ylabel("Predviđene Vrednosti")
plt.show()



