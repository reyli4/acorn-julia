import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
import pickle

from python.utils import project_path

warnings.filterwarnings("ignore")

home_types = [
    "mobile_home",
    "single-family_detached",
    "single-family_attached",
    "multi-family_with_2_-_4_units",
    "multi-family_with_5plus_units",
]

upgrades = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


class LoadPredictor:
    """
    Neural Network model for predicting NREL ResStock/ComStock
    electricity savings in New York based on hourly temperature,
    hour of day, and previous day's temperature using scikit-learn's MLPRegressor.
    """

    def __init__(
        self,
        temperature_col="T2C",
        target_col="savings_MW",
        time_col="time",
        hour_col="hour",
    ):
        """
        Initialize the LoadPredictor

        Parameters:
        -----------
        temperature_col : str
            Name of the temperature column
        target_col : str
            Name of the target load column
        time_col : str
            Name of the time column
        hour_col : str
            Name of the hour column
        """
        self.temperature_col = temperature_col
        self.target_col = target_col
        self.time_col = time_col
        self.models = {}
        self.scalers = {}
        self.results = {}

    def create_lag_features(self, df):
        """
        Create feature for previous calendar day's average load

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with time series data

        Returns:
        --------
        pandas.DataFrame
            Dataframe with previous day's average load feature added
        """
        df = df.copy()
        df = df.sort_values(self.time_col)

        # Calculate the average temperature for each calendar day
        daily_avg = df.groupby(df[self.time_col].dt.date)[self.temperature_col].mean()

        # Create a mapping of date to previous day's average
        prev_day_avg = daily_avg.shift(1)

        # Map the previous day's average to each hour
        df[f"{self.temperature_col}_prev_day_avg"] = df[self.time_col].dt.date.map(
            prev_day_avg
        )

        return df

    def prepare_features(self, df, additional_feature_cols=["hour"]):
        """
        Prepare features for neural network training

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with lag features

        Returns:
        --------
        tuple
            (X, y) where X is features and y is target
        """
        # Select features
        feature_cols = [
            self.temperature_col,
            f"{self.temperature_col}_prev_day_avg",
        ] + additional_feature_cols

        # Remove rows with NaN values (due to lag features)
        if self.target_col in df.columns:
            df_clean = df.dropna(subset=feature_cols + [self.target_col])
        else:
            df_clean = df.dropna(subset=feature_cols)

        if len(df_clean) == 0:
            return None, None

        X = df_clean[feature_cols].values
        if self.target_col in df.columns:
            y = df_clean[self.target_col].values
        else:
            y = None

        return X, y

    def create_neural_network(
        self,
        hidden_layer_sizes=(100, 100),
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
    ):
        """
        Create MLPRegressor neural network

        Parameters:
        -----------
        hidden_layer_sizes : tuple
            Tuple of hidden layer sizes
        alpha : float
            L2 regularization parameter
        learning_rate_init : float
            Initial learning rate
        max_iter : int
            Maximum number of iterations
        early_stopping : bool
            Whether to use early stopping
        validation_fraction : float
            Fraction of training data for validation (when early_stopping=True)

        Returns:
        --------
        sklearn.neural_network.MLPRegressor
            Configured neural network model
        """
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=15,
            random_state=42,
            batch_size="auto",
        )

        return model

    def train_test_split_timeseries(self, X, y, test_size=0.2):
        """
        Split time series data chronologically

        Parameters:
        -----------
        X : numpy.ndarray
            Features
        y : numpy.ndarray
            Target values
        test_size : float
            Proportion of data for testing

        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * (1 - test_size))

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        return X_train, X_test, y_train, y_test

    def evaluate_model(self, y_true, y_pred):
        """
        Calculate evaluation metrics

        Parameters:
        -----------
        y_true : numpy.ndarray
            True values
        y_pred : numpy.ndarray
            Predicted values

        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            "test_MAE": mae,
            "test_MSE": mse,
            "test_RMSE": rmse,
            "test_R2": r2,
            "test_MAPE": mape,
        }

    def fit_model(
        self,
        df,
        upgrade,
        home_type,
        hidden_layer_sizes=(100, 100),
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        validation_fraction=0.2,
        verbose=False,
    ):
        """
        Fit neural network model for specific upgrade and home_type

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        upgrade : int/str
            Upgrade number to filter on
        home_type : str
            Home type to filter on
        hidden_layer_sizes : tuple
            Tuple of hidden layer sizes
        alpha : float
            L2 regularization parameter
        learning_rate_init : float
            Initial learning rate
        max_iter : int
            Maximum number of iterations
        validation_fraction : float
            Fraction of training data for validation
        verbose : bool
            Whether to print detailed training info

        Returns:
        --------
        dict
            Results dictionary with model performance
        """
        # Filter data
        mask = (df["upgrade"] == upgrade) & (df["home_type"] == home_type)
        df_subset = df[mask].copy()

        if len(df_subset) < 100:  # Minimum data requirement
            print(
                f"Insufficient data for upgrade {upgrade}, home_type {home_type}: {len(df_subset)} samples"
            )
            return None

        print(
            f"Training model for upgrade {upgrade}, home_type {home_type} ({len(df_subset)} samples)"
        )

        # Create lag features
        df_subset = self.create_lag_features(df_subset)

        # Prepare features
        X, y = self.prepare_features(df_subset)

        if X is None or len(X) < 50:
            print(
                f"Insufficient clean data after feature engineering: {len(X) if X is not None else 0} samples"
            )
            return None

        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split_timeseries(X, y)

        # Scale features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # Create and train model
        model = self.create_neural_network(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            validation_fraction=validation_fraction,
        )

        # Train model
        if verbose:
            print(f"Training neural network with architecture: {hidden_layer_sizes}")

        model.fit(X_train_scaled, y_train_scaled)

        if verbose:
            print(f"Training completed in {model.n_iter_} iterations")
            if hasattr(model, "best_validation_score_"):
                print(f"Best validation score: {model.best_validation_score_:.4f}")

        # Make predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_train_pred_scaled = model.predict(X_train_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_train_pred = scaler_y.inverse_transform(
            y_train_pred_scaled.reshape(-1, 1)
        ).ravel()

        # Evaluate model
        metrics = self.evaluate_model(y_test, y_pred)

        # Store model and scalers
        model_key = f"{upgrade}_{home_type}"
        self.models[model_key] = model
        self.scalers[model_key] = {"X": scaler_X, "y": scaler_y}

        # Store results
        results = {
            "upgrade": upgrade,
            "home_type": home_type,
            "n_samples": len(df_subset),
            "X_train": X_train,
            "X_test": X_test,
            "metrics": metrics,
            "y_test_true": y_test,
            "y_test_pred": y_pred,
            "y_train_true": y_train,
            "y_train_pred": y_train_pred,
            "model_params": {
                "hidden_layer_sizes": hidden_layer_sizes,
                "alpha": alpha,
                "learning_rate_init": learning_rate_init,
                "n_iter": model.n_iter_,
            },
        }

        self.results[model_key] = results

        print(
            f"Model trained - R2: {metrics['test_R2']:.3f}, RMSE: {metrics['test_RMSE']:.2f} MW, Iterations: {model.n_iter_}"
        )

        return results

    def predict(self, X, upgrade, home_type):
        """
        Make predictions using trained model

        Parameters:
        -----------
        X : numpy.ndarray or pandas.DataFrame
            Input features [temperature, lag_daily_temperature, hour]
        upgrade : int/str
            Upgrade number
        home_type : str
            Home type

        Returns:
        --------
        numpy.ndarray
            Predictions
        """
        model_key = f"{upgrade}_{home_type}"

        if model_key not in self.models:
            raise ValueError(
                f"No trained model found for upgrade {upgrade}, home_type {home_type}"
            )

        model = self.models[model_key]
        scalers = self.scalers[model_key]

        # Scale input
        X_scaled = scalers["X"].transform(X)

        # Make prediction
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scalers["y"].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        return np.clip(y_pred, 0, None)

    def store_model(self, upgrade, home_type):
        """
        Store model and results
        """
        model_key = f"{upgrade}_{home_type}"

        model_store = {
            "model": self.models[model_key],
            "scaler": self.scalers[model_key],
            "results": self.results[model_key],
        }

        # Save model
        with open(
            f"{project_path}/data/load/resstock/models/{model_key}.pkl", "wb"
        ) as f:
            pickle.dump(model_store, f)

    def plot_results(self, upgrade, home_type, figsize=(10, 10)):
        """
        Plot training results and predictions

        Parameters:
        -----------
        upgrade : int/str
            Upgrade number
        home_type : str
            Home type
        figsize : tuple
            Figure size
        """
        model_key = f"{upgrade}_{home_type}"

        if model_key not in self.results:
            print(f"No results found for upgrade {upgrade}, home_type {home_type}")
            return

        results = self.results[model_key]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"Model Results: Upgrade {upgrade}, {home_type}", fontsize=14)

        # Training results
        axes[0, 0].scatter(results["y_train_true"], results["y_train_pred"], alpha=0.6)
        min_val = min(results["y_train_true"].min(), results["y_train_pred"].min())
        max_val = max(results["y_train_true"].max(), results["y_train_pred"].max())
        axes[0, 0].plot(
            [min_val, max_val], [min_val, max_val], ls="--", color="orange", lw=2
        )
        axes[0, 0].set_title("Training Results")
        axes[0, 0].set_xlabel("Actual Load (MW)")
        axes[0, 0].set_ylabel("Predicted Load (MW)")
        axes[0, 0].grid(True)

        # Add R² to the plot
        r2 = r2_score(results["y_train_true"], results["y_train_pred"])
        axes[0, 0].text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=axes[0, 0].transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Validation results
        axes[0, 1].scatter(results["y_test_true"], results["y_test_pred"], alpha=0.6)
        min_val = min(results["y_test_true"].min(), results["y_test_pred"].min())
        max_val = max(results["y_test_true"].max(), results["y_test_pred"].max())
        axes[0, 1].plot(
            [min_val, max_val], [min_val, max_val], ls="--", color="orange", lw=2
        )
        axes[0, 1].set_title("Validation Results")
        axes[0, 1].set_xlabel("Actual Load (MW)")
        axes[0, 1].set_ylabel("Predicted Load (MW)")
        axes[0, 1].grid(True)

        # Add R² to the plot
        r2 = r2_score(results["y_test_true"], results["y_test_pred"])
        axes[0, 1].text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=axes[0, 1].transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Temperature dependence
        axes[1, 0].scatter(results["X_train"][:, 0], results["y_train_pred"], alpha=0.6)
        axes[1, 0].set_title("Temperature Dependence")
        axes[1, 0].set_xlabel("Temperature [C]")
        axes[1, 0].set_ylabel("Predicted Load [MW]")
        axes[1, 0].grid(True)

        # Time series comparison (first 200 points)
        n_plot = min(200, len(results["y_train_true"]))
        axes[1, 1].plot(results["y_train_true"][:n_plot], label="Actual", alpha=0.7)
        axes[1, 1].plot(results["y_train_pred"][:n_plot], label="Predicted", alpha=0.7)
        axes[1, 1].set_title("Time Series Comparison (First 200 points)")
        axes[1, 1].set_xlabel("Time Step")
        axes[1, 1].set_ylabel("Load [MW]")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def get_model_info(self, upgrade, home_type):
        """
        Get detailed information about a trained model

        Parameters:
        -----------
        upgrade : int/str
            Upgrade number
        home_type : str
            Home type

        Returns:
        --------
        dict
            Model information
        """
        model_key = f"{upgrade}_{home_type}"

        if model_key not in self.models:
            return None

        model = self.models[model_key]
        results = self.results[model_key]

        info = {
            "architecture": results["model_params"]["hidden_layer_sizes"],
            "n_parameters": sum([layer.size for layer in model.coefs_])
            + sum([layer.size for layer in model.intercepts_]),
            "n_iterations": results["model_params"]["n_iter"],
            "alpha": results["model_params"]["alpha"],
            "learning_rate": results["model_params"]["learning_rate_init"],
            "final_loss": model.loss_,
            "convergence": model.n_iter_ < model.max_iter,
        }

        return info

    def summary_report(self):
        """
        Generate summary report of all trained models
        """
        if not self.results:
            print("No models trained yet.")
            return

        print("\n" + "=" * 90)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 90)

        summary_data = []
        for model_key, results in self.results.items():
            model_info = self.get_model_info(results["upgrade"], results["home_type"])

            summary_data.append(
                {
                    "Upgrade": results["upgrade"],
                    "Home Type": results["home_type"],
                    "Samples": results["n_samples"],
                    "Train Size": len(results["X_train"]),
                    "Test Size": len(results["X_test"]),
                    "R²": results["metrics"]["test_R2"],
                    "RMSE": results["metrics"]["test_RMSE"],
                    "MAE": results["metrics"]["test_MAE"],
                    "MAPE": results["metrics"]["test_MAPE"],
                    "Iterations": model_info["n_iterations"] if model_info else "N/A",
                    "Converged": model_info["convergence"] if model_info else "N/A",
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(["Upgrade", "Home Type"])

        print(summary_df.to_string(index=False, float_format="%.3f"))
        print("\n" + "=" * 90)

    def predict_future_loads(
        self,
        temp_file_path,
        temp_save_name,
        upgrades,
        home_types,
    ):
        """
        Make predictions for future loads
        """
        # Load new temperature data
        df_new = read_and_prepare_data(temp_file_path, read_resstock_data=False)

        # Create lag features
        df_new = self.create_lag_features(df_new)

        # Fill in missing values with average for day of year
        df_new[f"{self.temperature_col}_prev_day_avg"] = df_new.groupby("day_of_year")[
            f"{self.temperature_col}_prev_day_avg"
        ].transform(lambda x: x.fillna(x.mean()))

        # Prepare features
        X, _ = self.prepare_features(df_new)

        # Make predictions for all combinations
        for upgrade in upgrades:
            for home_type in home_types:
                model_key = f"{upgrade}_{home_type}"
                if model_key not in self.models:
                    raise ValueError(
                        f"No trained model found for upgrade {upgrade}, home_type {home_type}"
                    )
                # Make predictions
                y_pred = self.predict(X, upgrade, home_type)

                # Store results
                df_new["predicted_savings_MW"] = y_pred

                # Save results
                df_new.to_csv(
                    f"{project_path}/data/load/resstock/simulated/{temp_save_name}_{upgrade}_{home_type}.csv",
                    index=False,
                )


def train_load_prediction_models(
    df,
    upgrades,
    home_types,
    temperature_col="T2C",
    hidden_layer_sizes=(100, 50, 25),
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=500,
    plot_results=True,
    verbose=False,
    store_models=False,
):
    """
    Main function to train neural network models for load prediction using MLPRegressor

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with load data
    upgrades : list
        List of upgrade numbers to process
    home_types : list
        List of home types to process
    temperature_col : str
        Name of temperature column
    hidden_layer_sizes : tuple
        Tuple of hidden layer sizes for MLPRegressor
    alpha : float
        L2 regularization parameter
    learning_rate_init : float
        Initial learning rate
    max_iter : int
        Maximum number of iterations
    plot_results : bool
        Whether to plot results for each model
    verbose : bool
        Whether to print detailed training information

    Returns:
    --------
    LoadPredictor
        Trained LoadPredictor object
    """
    # Initialize predictor
    predictor = LoadPredictor(temperature_col=temperature_col)

    # Convert time column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    # Train models for each combination
    total_combinations = len(upgrades) * len(home_types)
    current_combination = 0

    print(f"Training {total_combinations} models with MLPRegressor...")
    print(f"Network architecture: {hidden_layer_sizes}")
    print(f"Regularization (alpha): {alpha}")
    print(f"Max iterations: {max_iter}")
    print("-" * 60)

    for upgrade in upgrades:
        for home_type in home_types:
            current_combination += 1
            print(f"\nProgress: {current_combination}/{total_combinations}")

            results = predictor.fit_model(
                df,
                upgrade,
                home_type,
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                verbose=verbose,
            )

            if results and plot_results:
                predictor.plot_results(upgrade, home_type)

            if store_models:
                predictor.store_model(upgrade, home_type)

    # Generate summary report
    predictor.summary_report()

    return predictor


def read_resstock_savings(home_type, upgrade):
    # Read
    upgrade_str = str(upgrade).zfill(2)
    df = pd.read_csv(
        f"{project_path}/data/nrel/resstock/up{upgrade_str}-nyiso-{home_type}.csv"
    )
    # Resample to hourly, convert to UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = (
        df.set_index("timestamp")
        .resample("h")[["out.electricity.total.energy_consumption.kwh.savings"]]
        .sum()
        .reset_index()
    )
    df["timestamp"] = df["timestamp"].dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    )
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    df = df.rename(columns={"timestamp": "time"})

    # Rename savings column
    df["savings_MW"] = (
        -df["out.electricity.total.energy_consumption.kwh.savings"] / 1000
    )

    df = df.drop(columns=["out.electricity.total.energy_consumption.kwh.savings"])

    # Add info
    df["upgrade"] = upgrade
    df["home_type"] = home_type

    return df


def read_and_prepare_data(temp_file_path, read_resstock_data=True, temp_varname="T2C"):
    """
    Preprocess temperature and load data from files.

    Parameters:
    -----------
    temp_file : str
        Path to temperature data file
    nrel_resstock_path : str
        Path to NREL electricity savings data files

    Returns:
    --------
    tuple
        Temperature and load DataFrames
    """
    try:
        # Temperature data
        temp_data = pd.read_csv(temp_file_path)
        # Ensure datetime columns are properly formatted
        temp_data["time"] = pd.to_datetime(temp_data["time"])

        # Add timezone indicator for temperature data
        temp_data["time"] = temp_data["time"].dt.tz_localize("UTC")

        # Average over zones if necessary
        temp_data = temp_data.groupby("time")[temp_varname].mean().reset_index()

        # Get NREL resstock data
        if read_resstock_data:
            # 2018 only
            temp_data = temp_data[temp_data["time"].dt.year == 2018]
            # Read resstock data
            resstock_data = pd.concat(
                [
                    read_resstock_savings(home_type, upgrade)
                    for home_type in home_types
                    for upgrade in upgrades
                ]
            )

            # Merge
            df = pd.merge(resstock_data, temp_data, on="time", how="inner")
        else:
            df = temp_data

        # Add relevant info
        df["hour"] = df["time"].dt.hour
        df["day_of_week"] = df["time"].dt.dayofweek
        df["day_of_year"] = df["time"].dt.dayofyear
        df["month"] = df["time"].dt.month
        df["year"] = df["time"].dt.year
        df["date"] = df["time"].dt.date

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
