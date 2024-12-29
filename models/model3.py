import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from load_data import load_data_campaign1, load_data_campaign2, load_data_campaign3  
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder

kampanja = "campaign3"

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

# Prebacivanje dometa u kategorije
def categorize_range(row):
    if row['remaining_range'] < 50:
        return 'short'
    elif 50 <= row['remaining_range'] <= 150:
        return 'medium'
    else:
        return 'long'

# Primjena kategorije na trening i test podatke
train_data['range_category'] = train_data.apply(categorize_range, axis=1)
test_data['range_category'] = test_data.apply(categorize_range, axis=1)

# Encoding kategorije (string to numeric)
label_encoder_range = LabelEncoder()
train_data['range_category'] = label_encoder_range.fit_transform(train_data['range_category'])
test_data['range_category'] = label_encoder_range.transform(test_data['range_category'])

# Odabir značajki za treniranje 
X_train = train_data[['speed', 'acceleration', 'completed_distance', 'alt', 'road_slope', 'wind', 'traffic_factor', 'occupancy', 'auxiliaries']]
y_train = train_data['range_category']
X_test = test_data[['speed', 'acceleration', 'completed_distance', 'alt', 'road_slope', 'wind', 'traffic_factor', 'occupancy', 'auxiliaries']]
y_test = test_data['range_category']

# Treniranje Random Forest Classifier modela
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predikcija
y_train_pred = rf_classifier.predict(X_train)
y_test_pred = rf_classifier.predict(X_test)

# Rezultat
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Rezultat
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Matrica konfuzije
conf_matrix = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder_range.classes_, yticklabels=label_encoder_range.classes_)
plt.title("Matrica konfuzije")
plt.xlabel("Predviđene kategorije")
plt.ylabel("Stvarne kategorije")
plt.show()

# Histogram predikcija vs stvarnih vrijednosti
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(y_test_pred, kde=True, color='blue', label="Predikcija (RF)", stat="density")
sns.histplot(y_test, kde=True, color='red', label="Stvarne vrijednosti", stat="density")
plt.title('Distribucija: Predikcija vs Stvarne vrijednosti (Random Forest)')
plt.legend()

plt.tight_layout()
plt.show()
