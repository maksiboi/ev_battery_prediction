import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from load_data import load_data

class RangeClassificationModel:
    """
    A class-based implementation of a Random Forest Classifier for range category prediction.
    """
    def __init__(self, campaign: str):
        """
        Initialize the model with data from the specified campaign.
        
        Args:
            campaign (str): The campaign name (e.g., "campaign1").
        """
        self.campaign = campaign
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.range_encoder = LabelEncoder()
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        
        self._load_and_prepare_data()
    
    def _load_and_prepare_data(self):
        """
        Load and preprocess the training and testing data.
        """
        train_data, test_data = load_data(self.campaign)
        
        if train_data is None or test_data is None:
            raise ValueError(f"Data for campaign {self.campaign} could not be loaded.")
        
        train_data, test_data = self._encode_categorical_features(train_data, test_data)
        train_data, test_data = self._categorize_range(train_data, test_data)
        
        features = ['speed', 'acceleration', 'completed_distance', 'alt', 'road_slope', 'wind', 'traffic_factor', 'occupancy', 'auxiliaries']
        self.X_train, self.y_train = train_data[features], train_data['range_category']
        self.X_test, self.y_test = test_data[features], test_data['range_category']
    
    def _encode_categorical_features(self, train_data, test_data):
        """
        Encode categorical features using LabelEncoder.
        """
        train_data['vehicle_id'] = self.label_encoder.fit_transform(train_data['vehicle_id'])
        train_vehicle_ids = set(train_data['vehicle_id'])
        
        self.label_encoder.classes_ = np.append(self.label_encoder.classes_, 'unknown')
        test_data['vehicle_id'] = test_data['vehicle_id'].apply(lambda x: x if x in train_vehicle_ids else 'unknown')
        test_data['vehicle_id'] = self.label_encoder.transform(test_data['vehicle_id'])
        
        return train_data, test_data
    
    def _categorize_range(self, train_data, test_data):
        """
        Categorize remaining range into 'short', 'medium', and 'long'.
        """
        def categorize(row):
            if row['remaining_range'] < 50:
                return 'short'
            elif 50 <= row['remaining_range'] <= 150:
                return 'medium'
            else:
                return 'long'
        
        train_data['range_category'] = train_data.apply(categorize, axis=1)
        test_data['range_category'] = test_data.apply(categorize, axis=1)
        
        train_data['range_category'] = self.range_encoder.fit_transform(train_data['range_category'])
        test_data['range_category'] = self.range_encoder.transform(test_data['range_category'])
        
        return train_data, test_data
    
    def train(self):
        """
        Train the Random Forest model.
        """
        self.rf_classifier.fit(self.X_train, self.y_train)
    
    def predict(self):
        """
        Generate predictions.
        """
        return self.rf_classifier.predict(self.X_test)
    
    def evaluate(self):
        """
        Evaluate the model using accuracy score and confusion matrix.
        """
        y_pred = self.predict()
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        return accuracy, conf_matrix
    
    def plot_confusion_matrix(self):
        """
        Plot confusion matrix.
        """
        _, conf_matrix = self.evaluate()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=self.range_encoder.classes_, yticklabels=self.range_encoder.classes_)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Category")
        plt.ylabel("Actual Category")
        plt.show()
    
    def save_model(self, filename: str):
        """
        Save the trained model.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self.rf_classifier, file)
        print(f"Model saved to {filename}")

# Example Usage
if __name__ == "__main__":
    model = RangeClassificationModel("campaign3")
    model.train()
    accuracy, _ = model.evaluate()
    print(f"Model Accuracy: {accuracy:.4f}")
    model.save_model("data/model_data/model3.pkl")
