from load_data import load_data_campaign3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Učitavanje podataka za Campaign 3
train_data, test_data = load_data_campaign3()

if train_data is not None and test_data is not None:
    features = ['state_of_charge', 'speed', 'acceleration', 'completed_distance', 
                'alt', 'road_slope', 'wind', 'traffic_factor', 'occupancy', 'auxiliaries']

    def classify_range(distance):
        if distance < 50:
            return 0
        elif 50 <= distance < 150:
            return 1
        else:
            return 2

    train_data['range_category'] = train_data['completed_distance'].apply(classify_range)
    test_data['range_category'] = test_data['completed_distance'].apply(classify_range)

    X_train = train_data[features]
    y_train = train_data['range_category']
    X_test = test_data[features]
    y_test = test_data['range_category']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Točnost modela:", accuracy_score(y_test, y_pred))
    print("Izvještaj o klasifikaciji:\n", classification_report(y_test, y_pred))