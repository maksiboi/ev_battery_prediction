import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from load_data import load_data

class EnergyConsumptionModel:
    """
    A class-based implementation of Random Forest and Gradient Boosting models for energy consumption prediction.
    """
    def __init__(self, campaign: str):
        """
        Initialize the model with data from the specified campaign.
        
        Args:
            campaign (str): The campaign name (e.g., "campaign1").
        """
        self.campaign = campaign
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
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
        
        features = ['speed', 'acceleration', 'road_slope', 'auxiliaries', 'traffic_factor', 'wind']
        self.X_train, self.y_train = train_data[features], train_data['total_energy_consumed_wh']
        self.X_test, self.y_test = test_data[features], test_data['total_energy_consumed_wh']
    
    def _encode_categorical_features(self, train_data, test_data):
        """
        Encode categorical features using LabelEncoder.
        """
        train_data['vehicle_id'] = self.label_encoder.fit_transform(train_data['vehicle_id'])
        train_vehicle_ids = set(train_data['vehicle_id'])
        
        # Handle unknown categories in test set
        self.label_encoder.classes_ = np.append(self.label_encoder.classes_, 'unknown')
        test_data['vehicle_id'] = test_data['vehicle_id'].apply(lambda x: x if x in train_vehicle_ids else 'unknown')
        test_data['vehicle_id'] = self.label_encoder.transform(test_data['vehicle_id'])
        
        return train_data, test_data
    
    def train(self):
        """
        Train both models.
        """
        self.rf_model.fit(self.X_train, self.y_train)
        self.gb_model.fit(self.X_train, self.y_train)
    
    def predict(self):
        """
        Generate predictions for both models.
        """
        return self.rf_model.predict(self.X_test), self.gb_model.predict(self.X_test)
    
    def evaluate(self):
        """
        Evaluate both models using MSE and R² Score.
        """
        rf_pred, gb_pred = self.predict()
        metrics = {
            "RandomForest": {
                "MSE": mean_squared_error(self.y_test, rf_pred),
                "R2": r2_score(self.y_test, rf_pred)
            },
            "GradientBoosting": {
                "MSE": mean_squared_error(self.y_test, gb_pred),
                "R2": r2_score(self.y_test, gb_pred)
            }
        }
        return metrics
    
    def plot_predictions(self):
        """
        Plot prediction distributions.
        """
        rf_pred, gb_pred = self.predict()
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(rf_pred, kde=True, color='blue', label="Prediction (RF)", stat="density")
        sns.histplot(self.y_test, kde=True, color='red', label="Actual Values", stat="density")
        plt.title('Distribution: Prediction vs Actual (Random Forest)')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        sns.histplot(gb_pred, kde=True, color='blue', label="Prediction (GB)", stat="density")
        sns.histplot(self.y_test, kde=True, color='red', label="Actual Values", stat="density")
        plt.title('Distribution: Prediction vs Actual (Gradient Boosting)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, rf_filename: str, gb_filename: str):
        """
        Save the trained Random Forest and Gradient Boosting models separately.
        """
        with open(rf_filename, 'wb') as rf_file:
            pickle.dump(self.rf_model, rf_file)
        print(f"Random Forest model saved to {rf_filename}")
        
        with open(gb_filename, 'wb') as gb_file:
            pickle.dump(self.gb_model, gb_file)
        print(f"Gradient Boosting model saved to {gb_filename}")

# Example Usage
if __name__ == "__main__":
    campaign_model = EnergyConsumptionModel("campaign2")
    campaign_model.train()
    metrics = campaign_model.evaluate()
    print("Model Evaluation:", metrics)
    campaign_model.save_model("data/model_data/model2/rf_model.pkl", "data/model_data/model2/gb_model.pkl")