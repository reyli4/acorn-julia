import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import timedelta
import pickle
import os

class MultiZoneLoadPredictor:
    """
    Predicts electricity loads for multiple zones simultaneously using
    a multi-output regression model.
    """
    
    def __init__(self, model=None, model_type="random_forest"):
        """
        Initialize the multi-zone load predictor with a specified model.
        
        Parameters:
        -----------
        model : scikit-learn model, optional
            The ML model to use for prediction.
        model_type : str, optional
            Type of model to use if model is None. Options: "random_forest", "ridge"
        """
        if model is not None:
            self.model = model
            self.model_type = "custom"
        else:
            if model_type == "random_forest":
                # RandomForestRegressor supports multi-output natively
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")
            self.model_type = model_type
            
        self.scaler = StandardScaler()
        self.zones = None
        self.zone_columns = None
        
    def preprocess_data(self, temp_data, load_data, temp_varname='T2C'):
        """
        Preprocess the temperature and load data for all zones.
        
        Parameters:
        -----------
        temp_data : DataFrame
            Temperature data with columns 'zone', 'time', temp_varname
        load_data : DataFrame
            Load data with columns 'time', 'zone', 'load_MW'
            
        Returns:
        --------
        DataFrame
            Preprocessed data with features and multi-zone targets
        """
        # Convert time columns to datetime if they aren't already
        temp_data['time'] = pd.to_datetime(temp_data['time'])
        load_data['time'] = pd.to_datetime(load_data['time'])
        
        # Rename columns for consistency
        temp_data = temp_data.rename(columns={'time': 'datetime'})
        load_data = load_data.rename(columns={'time': 'datetime'})
        
        # Store unique zones
        self.zones = sorted(load_data['zone'].unique())
        
        # Pivot the load data to have one column per zone
        load_pivot = load_data.pivot_table(
            index='datetime', 
            columns='zone', 
            values='load_MW',
        ).reset_index()
        
        # Store zone column names
        self.zone_columns = load_pivot.columns[1:].tolist()
        
        # Create datetime-indexed dataframe for joining
        temp_data_wide = temp_data.pivot_table(
            index='datetime',
            columns='zone',
            values=temp_varname,
        )
        
        # Rename temperature columns to avoid confusion with load columns
        temp_columns = {zone: f'{temp_varname}_{zone}' for zone in temp_data_wide.columns}
        temp_data_wide = temp_data_wide.rename(columns=temp_columns)
        temp_data_wide = temp_data_wide.reset_index()
        
        # Merge load and temperature data
        data = pd.merge(load_pivot, temp_data_wide, on='datetime', how='inner')
        
        # Extract temporal features
        data['day_of_week'] = data['datetime'].dt.dayofweek
        data['day_of_year'] = data['datetime'].dt.dayofyear
        data['hour'] = data['datetime'].dt.hour
        data['month'] = data['datetime'].dt.month
        data['year'] = data['datetime'].dt.year
        
        # Calculate previous day's average load for each zone
        data = data.sort_values('datetime')
        data['date'] = data['datetime'].dt.date
        
        # Create previous day average load features for each zone
        for zone in self.zones:
            # Group by date and calculate daily average for each zone
            zone_daily_avg = data.groupby('date')[zone].mean().reset_index()
            zone_daily_avg['prev_date'] = pd.to_datetime(zone_daily_avg['date']) + timedelta(days=1)
            zone_daily_avg = zone_daily_avg.rename(columns={zone: f'prev_day_avg_{zone}'})
            
            # Merge with previous day's average
            data['date'] = pd.to_datetime(data['date'])
            data = pd.merge(
                data, 
                zone_daily_avg[['prev_date', f'prev_day_avg_{zone}']],
                left_on='date', 
                right_on='prev_date', 
                how='left'
            )
            data = data.drop(columns=['prev_date'], errors='ignore')
            
            # Fill missing values for the first day with the zone's mean
            if data[f'prev_day_avg_{zone}'].isna().any():
                data[f'prev_day_avg_{zone}'] = data[f'prev_day_avg_{zone}'].fillna(data[zone].mean())
        
        # Drop unnecessary columns
        data = data.drop(columns=['prev_date'], errors='ignore')
        
        return data
    
    def prepare_features_target(self, data, temp_varname='T2C'):
        """
        Prepare features and multi-zone target variables from preprocessed data.
        
        Parameters:
        -----------
        data : DataFrame
            Preprocessed data
            
        Returns:
        --------
        X : DataFrame
            Feature matrix
        y : DataFrame
            Multi-zone target matrix
        """
        # Basic temporal features
        base_features = ['day_of_week', 'day_of_year']
        
        # Temperature features for each zone
        temp_features = [col for col in data.columns if col.startswith(f'{temp_varname}_')]
        
        # Previous day average load features
        prev_load_features = [col for col in data.columns if col.startswith('prev_day_avg_')]
        
        # Combine all features
        feature_cols = base_features + temp_features + prev_load_features
        
        # Select features and targets
        X = data[feature_cols]
        y = data[self.zone_columns]
        
        return X, y
    
    def train(self, temp_data, load_data, test_split=0.2, temp_varname='T2C', random_state=42):
        """
        Train the model to predict loads for all zones simultaneously.
        
        Parameters:
        -----------
        temp_data : DataFrame
            Temperature data
        load_data : DataFrame
            Load data
        test_split : float, optional
            Proportion of data to use for testing. Default is 0.2.
        random_state : int, optional
            Random state for reproducibility.
            
        Returns:
        --------
        dict
            Training results including metrics
        """
        # Preprocess data
        data = self.preprocess_data(temp_data, load_data)
        
        # Prepare features and target
        X, y = self.prepare_features_target(data)
        
        # Sort data chronologically for time-series split
        data = data.sort_values('datetime')
        X = X.loc[data.index]
        y = y.loc[data.index]
        
        # Train/test split - using chronological split for time series data
        # Train/test split - using chronological split for time series data
        if type(test_split)is float:
            split_idx = int(len(X) * (1 - test_split))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        elif type(test_split)is list:
            split_idx = data["year"].isin(test_split)
            X_train, X_test = X[~split_idx], X[split_idx]
            y_train, y_test = y[~split_idx], y[split_idx]
        else:
            print("Invalid split type")
            return None
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Get predictions
        y_pred_test = self.predict(X_test_scaled)
        y_pred_train = self.model.predict(X_train_scaled)
        
        # Calculate metrics for each zone
        metrics = {}
        for i, zone in enumerate(self.zone_columns):
            zone_metrics = {
                'rmse_train': np.sqrt(mean_squared_error(y_train[zone], y_pred_train[:, i])),
                'mae_train': mean_absolute_error(y_train[zone], y_pred_train[:, i]),
                'r2_train': r2_score(y_train[zone], y_pred_train[:, i]),
                'rmse_test': np.sqrt(mean_squared_error(y_test[zone], y_pred_test[:, i])),
                'mae_test': mean_absolute_error(y_test[zone], y_pred_test[:, i]),
                'r2_test': r2_score(y_test[zone], y_pred_test[:, i])
            }
            metrics[zone] = zone_metrics
        
        # Calculate overall metrics
        overall_metrics = {
            'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'mae_train': mean_absolute_error(y_train, y_pred_train),
            'r2_train': r2_score(y_train, y_pred_train),
            'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae_test': mean_absolute_error(y_test, y_pred_test),
            'r2_test': r2_score(y_test, y_pred_test)
        }
        metrics['overall'] = overall_metrics

        # Store test results and metrics for visualization
        self.test_results = {
            'y_true': y_test,
            'y_pred': y_pred_test,
            'test_datetimes': data[split_idx]['datetime'].to_numpy(),
            'test_temps': data[split_idx][[f"{temp_varname}_{zone}" for zone in self.zone_columns]].to_numpy(),
            'metrics': metrics,
            'feature_names': X.columns.tolist()
        }
        self.metrics = metrics
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions for all zones and ensure they are non-negative.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix
            
        Returns:
        --------
        array
            Non-negative predictions for all zones
        """
        # Check if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            # Scale features
            X_scaled = self.scaler.transform(X)
        else:
            # Assume X is already scaled
            X_scaled = X
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Ensure non-negative predictions
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.clip(lower=0)
        else:
            predictions = np.maximum(0, predictions)
        
        return predictions
    
    def evaluate(self):
        """
        Return evaluation metrics for all zones.
        
        Returns:
        --------
        dict
            Evaluation metrics by zone
        """
        if not hasattr(self, 'test_results'):
            raise ValueError("Model has not been trained yet")
        
        return self.test_results['metrics']
    
    def plot_results(self, zone, filepath=None):
        """
        Plot actual vs predicted values for specified zones.
        
        Parameters:
        -----------
        zones : list of str, optional
            zones to plot. If None, plots all zones.
        """
        if not hasattr(self, 'test_results'):
            raise ValueError("Model has not been trained yet")
        
        # Get required data
        zone_idx = self.zone_columns.index(zone)
        y_true = self.test_results['y_true'][zone]
        y_pred = self.test_results['y_pred'][:, zone_idx]
        temp = self.test_results['test_temps'][:, zone_idx]
        # datetimes = self.test_results['test_datetimes']
        
        fig, axs = plt.subplots(3,1, figsize=(8,8))
        fig.suptitle(f"Actual vs Predicted Load for Zone {zone}")

        # Timeseries plot
        ax=axs[0]
        ax.plot(y_true, label='Actual')
        ax.plot(y_pred, label='Prediction')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Load (MW)')
        ax.grid(alpha=0.4)
        ax.legend()

        # Temperature scatter plot
        ax=axs[1]
        ax.scatter(temp, y_true, label='Actual', s=5, alpha=0.5)
        ax.scatter(temp, y_pred, label='Prediction', s=5, alpha=0.5)
        ax.set_xlabel('Zonal temperature (C)')
        ax.set_ylabel('Load (MW)')
        ax.grid(alpha=0.4)
        ax.legend()

        # Scatter plot
        ax=axs[2]
        ax.scatter(y_true, y_pred, s=5, alpha=0.5)
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', color='black')
        ax.set_xlabel('Actual load (MW)')
        ax.set_ylabel('Predicted load (MW)')
        ax.grid(alpha=0.4)

        plt.tight_layout()

        if filepath is not None:
            plt.savefig(filepath, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_feature_importance(self):
        """
        Plot feature importance for all zones (if the model supports it).
        """
        if not hasattr(self, 'test_results'):
            raise ValueError("Model has not been trained yet")
        
        # Check if the model has feature_importances_ attribute
        if hasattr(self.model, 'feature_importances_'):
            # Direct access for RandomForestRegressor
            importances = self.model.feature_importances_
            feature_names = self.test_results['feature_names']
        elif self.model_type == "random_forest":
            # Direct access for RandomForestRegressor
            importances = self.model.feature_importances_
            feature_names = self.test_results['feature_names']
        elif hasattr(self.model, 'estimators_'):
            # For MultiOutputRegressor, get average importance across all outputs
            all_importances = []
            for estimator in self.model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    all_importances.append(estimator.feature_importances_)
            
            if all_importances:
                importances = np.mean(all_importances, axis=0)
                feature_names = self.test_results['feature_names']
            else:
                print("Model doesn't support feature importance visualization")
                return
        else:
            print("Model doesn't support feature importance visualization")
            return
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        sorted_importances = importances[indices]
        sorted_features = [feature_names[i] for i in indices]
        
        # Create a plot
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance (All zones)')
        plt.bar(range(len(sorted_importances)), sorted_importances)
        plt.xticks(range(len(sorted_importances)), sorted_features, rotation=90)
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath="multi_zone_model.pkl"):
        """
        Save the trained model and related data.
        
        Parameters:
        -----------
        filepath : str, optional
            The filepath to save the model to.
        """
        if not hasattr(self, 'test_results'):
            raise ValueError("Model has not been trained yet")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'zones': self.zones,
            'zone_columns': self.zone_columns,
            'test_results': self.test_results,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="multi_zone_model.pkl"):
        """
        Load a trained model and related data.
        
        Parameters:
        -----------
        filepath : str, optional
            The filepath to load the model from.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.zones = model_data['zones']
        self.zone_columns = model_data['zone_columns']
        self.test_results = model_data['test_results']
        self.metrics = model_data['metrics']
        self.model_type = model_data.get('model_type', 'custom')
        
        print(f"Model loaded from {filepath}")
    
    def get_zone_prediction(self, X, zone):
        """
        Get predictions for a specific zone.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix
        zone : str
            zone to get predictions for
            
        Returns:
        --------
        array
            Predictions for the specified zone
        """
        if zone not in self.zone_columns:
            raise ValueError(f"zone {zone} not found in trained model")
        
        # Get zone index
        zone_idx = self.zone_columns.index(zone)
        
        # Get predictions for all zones
        all_predictions = self.predict(X)
        
        # Extract predictions for specified zone
        return all_predictions[:, zone_idx]


def load_and_prepare_data(temp_file, load_file):
    """
    Load temperature and load data from files.
    
    Parameters:
    -----------
    temp_file : str
        Path to temperature data file
    load_file : str
        Path to load data file
        
    Returns:
    --------
    tuple
        Temperature and load DataFrames
    """
    try:
        # Load data
        temp_data = pd.read_csv(temp_file)
        load_data = pd.read_csv(load_file)
        
        # Ensure datetime columns are properly formatted
        temp_data['time'] = pd.to_datetime(temp_data['time'])
        load_data['time'] = pd.to_datetime(load_data['time'])

        # Convert load data to UTC
        load_data['time'] = load_data['time'].dt.tz_localize('America/New_York', ambiguous='NaT')
        load_data['time'] = load_data['time'].dt.tz_convert('UTC')

        # Add timezone indicator for temperature data
        temp_data['time'] = temp_data['time'].dt.tz_localize('UTC')
        
        return temp_data, load_data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise