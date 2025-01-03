import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from load_data import load_data
from sklearn.preprocessing import LabelEncoder

class RandomForestModel:
    """
    A class-based implementation of a Random Forest model for regression.
    """
    def __init__(self, campaign: str):
        """
        Initialize the model with data from the specified campaign.
        
        Args:
            campaign (str): The campaign name (e.g., "campaign1").
        """
        self.campaign = campaign
        self.model = RandomForestRegressor(
            n_estimators=50,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42
        )
        self.vehicle_id_encoder = LabelEncoder()
        self.trip_plan_encoder = LabelEncoder()
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        
        self._load_and_prepare_data()
    
    def _load_and_prepare_data(self):
        """
        Load and preprocess the training and testing data.
        """
        train_data, test_data = load_data(self.campaign)
        
        if train_data is None or test_data is None:
            raise ValueError(f"Data for campaign {self.campaign} could not be loaded.")
        
        # Encode categorical columns
        train_data, test_data = self._encode_categorical_features(train_data, test_data)
        
        # Select features and target
        remaining_features = [col for col in train_data.columns if col != "remaining_range"]
        self.X_train = train_data[remaining_features].dropna()
        self.y_train = train_data.loc[self.X_train.index, "remaining_range"]
        
        self.X_test = test_data[remaining_features].dropna()
        self.y_test = test_data.loc[self.X_test.index, "remaining_range"]
    
    def _encode_categorical_features(self, train_data, test_data):
        """
        Encode categorical features using LabelEncoder.
        
        Args:
            train_data (pd.DataFrame): Training data.
            test_data (pd.DataFrame): Testing data.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Encoded training and testing data.
        """
        train_data['vehicle_id'] = self.vehicle_id_encoder.fit_transform(train_data['vehicle_id'])
        test_data['vehicle_id'] = self.vehicle_id_encoder.transform(test_data['vehicle_id'])
        
        train_data['trip_plan'] = self.trip_plan_encoder.fit_transform(train_data['trip_plan'])
        test_data['trip_plan'] = self.trip_plan_encoder.transform(test_data['trip_plan'])
        
        return train_data, test_data
    
    def train(self):
        """
        Train the Random Forest model.
        """
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self):
        """
        Generate predictions for both training and testing data.
        
        Returns:
            tuple: (y_train_pred, y_test_pred) - Predictions for train and test data.
        """
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        return y_train_pred, y_test_pred
    
    def evaluate(self):
        """
        Evaluate the model using Mean Squared Error (MSE) and R² Score.
        
        Returns:
            dict: Evaluation metrics for both training and testing data.
        """
        y_train_pred, y_test_pred = self.predict()
        
        metrics = {
            "train_mse": mean_squared_error(self.y_train, y_train_pred),
            "train_r2": r2_score(self.y_train, y_train_pred),
            "test_mse": mean_squared_error(self.y_test, y_test_pred),
            "test_r2": r2_score(self.y_test, y_test_pred)
        }
        return metrics
    
    def plot_feature_importance(self):
        """
        Plot the feature importance based on the trained model.
        """
        feature_importances = self.model.feature_importances_
        features = self.X_train.columns
        
        plt.figure(figsize=(10, 5))
        plt.barh(features, feature_importances, color='skyblue')
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance in RandomForest Model")
        plt.show()
    
    def save_model(self, filename: str):
        """
        Save the trained model as a .pkl file.
        
        Args:
            filename (str): Path to save the model.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"Model saved to {filename}")


if __name__ == "__main__":
    campaign_model = RandomForestModel("campaign1")
    campaign_model.train()
    metrics = campaign_model.evaluate()
    print("Model Evaluation:", metrics)
    campaign_model.save_model("data/model_data/model1.pkl")
