# src/python/building_elec_model.py

import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
import pickle

# --- Project root discovery (env var > utils.project_path > filesystem fallback) ---
try:
    # optional: if your repo defines python.utils.project_path, we can reuse it
    from python.utils import project_path as _DEFAULT_PROJECT_PATH  # may be a str
except Exception:
    _DEFAULT_PROJECT_PATH = None

PROJECT_PATH = Path(
    os.getenv("ACORN_PROJECT_PATH", _DEFAULT_PROJECT_PATH)
    if _DEFAULT_PROJECT_PATH is not None
    else os.getenv("ACORN_PROJECT_PATH", str(Path(__file__).resolve().parents[2]))
).resolve()

warnings.filterwarnings("ignore")

building_types_dict = {
    "resstock": [
        "mobile_home",
        "single-family_detached",
        "single-family_attached",
        "multi-family_with_2_-_4_units",
        "multi-family_with_5plus_units",
    ],
    "comstock": [
        "outpatient",
        "hospital",
        "largeoffice",
        "largehotel",
        "smalloffice",
        "retailstandalone",
        "warehouse",
        "secondaryschool",
        "retailstripmall",
        "smallhotel",
        "primaryschool",
        "quickservicerestaurant",
        "mediumoffice",
        "fullservicerestaurant",
    ],
}

building_type_meta_map = {
    "mobile_home": "Mobile Home",
    "single-family_detached": "Single-Family Detached",
    "single-family_attached": "Single-Family Attached",
    "multi-family_with_2_-_4_units": "Multi-Family 2 - 4 Units",
    "multi-family_with_5plus_units": "Multi-Family with 5+ Units",
    "outpatient": "Outpatient",
    "hospital": "Hospital",
    "largeoffice": "LargeOffice",
    "largehotel": "LargeHotel",
    "smalloffice": "SmallOffice",
    "retailstandalone": "RetailStandalone",
    "warehouse": "Warehouse",
    "secondaryschool": "SecondarySchool",
    "retailstripmall": "RetailStripMall",
    "smallhotel": "SmallHotel",
    "primaryschool": "PrimarySchool",
    "quickservicerestaurant": "Quick ServiceRestaurant",
    "mediumoffice": "MediumOffice",
    "fullservicerestaurant": "FullServiceRestaurant",
}

upgrades_dict = {
    "resstock": np.arange(1, 17),
    "comstock": np.arange(1, 31),
}


class LoadPredictor:
    """
    Neural Network model for predicting NREL ResStock/ComStock
    electricity savings in New York based on hourly temperature,
    hour of day, and previous day's temperature using scikit-learn's MLPRegressor.
    """

    def __init__(
        self,
        stock_type,
        temperature_col="T2C",
        target_col="savings_MW",
        time_col="time",
        hour_col="hour",
    ):
        """
        Initialize the LoadPredictor
        """
        self.stock_type = stock_type
        self.temperature_col = temperature_col
        self.target_col = target_col
        self.time_col = time_col
        self.models = {}
        self.scalers = {}
        self.results = {}

    def create_lag_features(self, df):
        """
        Create feature for previous calendar day's average load
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
        building_type,
        hidden_layer_sizes=(100, 100),
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        validation_fraction=0.2,
        verbose=False,
    ):
        """
        Fit neural network model for specific upgrade and building_type
        """
        # Filter data
        mask = (df["upgrade"] == upgrade) & (df["building_type"] == building_type)
        df_subset = df[mask].copy()

        if len(df_subset) < 100:  # Minimum data requirement
            print(
                f"Insufficient data for upgrade {upgrade}, building_type {building_type}: {len(df_subset)} samples"
            )
            return None

        print(
            f"Training model for upgrade {upgrade}, building_type {building_type} ({len(df_subset)} samples)"
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

        if verbose:
            print(f"Training neural network with architecture: {hidden_layer_sizes}")

        model.fit(X_train_scaled, y_train_scaled)

        if verbose:
            print(f"Training completed in {model.n_iter_} iterations")
            if hasattr(model, "best_validation_score_"):
                print(f"Best validation score: {model.best_validation_score_:.4f}")

        # Predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_train_pred_scaled = model.predict(X_train_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_train_pred = scaler_y.inverse_transform(
            y_train_pred_scaled.reshape(-1, 1)
        ).ravel()

        # Evaluate
        metrics = self.evaluate_model(y_test, y_pred)

        # Store
        model_key = f"{upgrade}_{building_type}"
        self.models[model_key] = model
        self.scalers[model_key] = {"X": scaler_X, "y": scaler_y}
        self.results[model_key] = {
            "upgrade": upgrade,
            "building_type": building_type,
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

        print(
            f"Model trained - R2: {metrics['test_R2']:.3f}, RMSE: {metrics['test_RMSE']:.2f} MW, Iterations: {model.n_iter_}"
        )
        return self.results[model_key]

    def predict(self, X, upgrade, building_type):
        """
        Make predictions using trained model
        """
        model_key = f"{upgrade}_{building_type}"
        if model_key not in self.models:
            raise ValueError(
                f"No trained model found for upgrade {upgrade}, building_type {building_type}"
            )
        model = self.models[model_key]
        scalers = self.scalers[model_key]
        X_scaled = scalers["X"].transform(X)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scalers["y"].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return np.clip(y_pred, 0, None)

    def store_model(self, upgrade, building_type):
        """
        Store model and results to disk (ensures directory exists)
        """
        model_key = f"{upgrade}_{building_type}"
        model_store = {
            "model": self.models[model_key],
            "scaler": self.scalers[model_key],
            "results": self.results[model_key],
        }
        model_dir = PROJECT_PATH / "data" / "load" / self.stock_type / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / f"{model_key}.pkl", "wb") as f:
            pickle.dump(model_store, f)

    def plot_results(self, upgrade, building_type, figsize=(10, 10)):
        """
        Plot training results and predictions
        """
        model_key = f"{upgrade}_{building_type}"
        if model_key not in self.results:
            print(
                f"No results found for upgrade {upgrade}, building_type {building_type}"
            )
            return

        results = self.results[model_key]
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"Model Results: Upgrade {upgrade}, {building_type}", fontsize=14)

        # Training scatter
        axes[0, 0].scatter(results["y_train_true"], results["y_train_pred"], alpha=0.6)
        min_val = min(results["y_train_true"].min(), results["y_train_pred"].min())
        max_val = max(results["y_train_true"].max(), results["y_train_pred"].max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], ls="--", lw=2)
        axes[0, 0].set_title("Training Results")
        axes[0, 0].set_xlabel("Actual Load (MW)")
        axes[0, 0].set_ylabel("Predicted Load (MW)")
        axes[0, 0].grid(True)
        r2 = r2_score(results["y_train_true"], results["y_train_pred"])
        axes[0, 0].text(
            0.05, 0.95, f"R² = {r2:.3f}", transform=axes[0, 0].transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Validation scatter
        axes[0, 1].scatter(results["y_test_true"], results["y_test_pred"], alpha=0.6)
        min_val = min(results["y_test_true"].min(), results["y_test_pred"].min())
        max_val = max(results["y_test_true"].max(), results["y_test_pred"].max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], ls="--", lw=2)
        axes[0, 1].set_title("Validation Results")
        axes[0, 1].set_xlabel("Actual Load (MW)")
        axes[0, 1].set_ylabel("Predicted Load (MW)")
        axes[0, 1].grid(True)
        r2 = r2_score(results["y_test_true"], results["y_test_pred"])
        axes[0, 1].text(
            0.05, 0.95, f"R² = {r2:.3f}", transform=axes[0, 1].transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Temperature dependence (train set)
        axes[1, 0].scatter(results["X_train"][:, 0], results["y_train_pred"], alpha=0.6)
        axes[1, 0].set_title("Temperature Dependence")
        axes[1, 0].set_xlabel("Temperature [C]")
        axes[1, 0].set_ylabel("Predicted Load [MW]")
        axes[1, 0].grid(True)

        # Short time series
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

    def get_model_info(self, upgrade, building_type):
        """
        Get detailed information about a trained model
        """
        model_key = f"{upgrade}_{building_type}"
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
            model_info = self.get_model_info(
                results["upgrade"], results["building_type"]
            )
            summary_data.append(
                {
                    "Upgrade": results["upgrade"],
                    "Home Type": results["building_type"],
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

        summary_df = pd.DataFrame(summary_data).sort_values(["Upgrade", "Home Type"])
        print(summary_df.to_string(index=False, float_format="%.3f"))
        print("\n" + "=" * 90)

    def predict_future_loads(
        self,
        temp_file_path,
        temp_save_name,
        upgrades,
        building_types,
    ):
        """
        Make predictions for future loads and save to repo-relative paths
        """
        # Load new temperature data
        df_new = read_and_prepare_data(
            temp_file_path=temp_file_path,
            stock_type=self.stock_type,
            read_stock_data=False,
            building_types=building_types,
            upgrades=upgrades,
        )

        # Create lag features
        df_new = self.create_lag_features(df_new)

        # Fill missing lag values with average for day of year
        df_new[f"{self.temperature_col}_prev_day_avg"] = df_new.groupby("day_of_year")[
            f"{self.temperature_col}_prev_day_avg"
        ].transform(lambda x: x.fillna(x.mean()))

        # Prepare features
        X, _ = self.prepare_features(df_new)

        save_dir = PROJECT_PATH / "data" / "load" / self.stock_type / "simulated" / "state_wide"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Predict for all combinations
        for upgrade in upgrades:
            for building_type in building_types:
                model_key = f"{upgrade}_{building_type}"
                if model_key not in self.models:
                    raise ValueError(
                        f"No trained model found for upgrade {upgrade}, building_type {building_type}"
                    )
                y_pred = self.predict(X, upgrade, building_type)
                df_out = df_new.copy()
                df_out["predicted_savings_MW"] = y_pred
                df_out.to_csv(save_dir / f"{temp_save_name}_{upgrade}_{building_type}.csv", index=False)


def train_load_prediction_models(
    stock_type,
    df,
    upgrades,
    building_types,
    temperature_col="T2C",
    hidden_layer_sizes=(100, 100),
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=500,
    plot_results=True,
    verbose=False,
    store_models=False,
):
    """
    Train neural network models for load prediction using MLPRegressor
    """
    predictor = LoadPredictor(temperature_col=temperature_col, stock_type=stock_type)

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    total_combinations = len(upgrades) * len(building_types)
    current_combination = 0

    print(f"Training {total_combinations} models with MLPRegressor...")
    print(f"Network architecture: {hidden_layer_sizes}")
    print(f"Regularization (alpha): {alpha}")
    print(f"Max iterations: {max_iter}")
    print("-" * 60)

    for upgrade in upgrades:
        for building_type in building_types:
            current_combination += 1
            print(f"\nProgress: {current_combination}/{total_combinations}")

            results = predictor.fit_model(
                df,
                upgrade,
                building_type,
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                verbose=verbose,
            )

            if results and plot_results:
                predictor.plot_results(upgrade, building_type)

            if store_models:
                predictor.store_model(upgrade, building_type)

    predictor.summary_report()
    return predictor


def read_savings(stock_type, building_type, upgrade):
    """
    Read a single ResStock/ComStock CSV for one upgrade + building_type
    from repo-relative path.
    """
    upgrade_str = str(upgrade).zfill(2)
    csv_path = PROJECT_PATH / "data" / "nrel" / stock_type / f"up{upgrade_str}-nyiso-{building_type}.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return None

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

    # Convert savings to MW (negative savings -> positive reduction)
    df["savings_MW"] = -df["out.electricity.total.energy_consumption.kwh.savings"] / 1000
    df = df.drop(columns=["out.electricity.total.energy_consumption.kwh.savings"])

    # Add identifiers
    df["upgrade"] = upgrade
    df["building_type"] = building_type
    return df


def read_and_prepare_data(
    temp_file_path,
    stock_type,
    building_types,
    upgrades,
    read_stock_data=True,
    temp_varname="T2C",
):
    """
    Preprocess temperature and optional stock data, return merged DataFrame.
    """
    try:
        # Temperature data
        temp_data = pd.read_csv(temp_file_path)
        temp_data["time"] = pd.to_datetime(temp_data["time"])
        temp_data["time"] = temp_data["time"].dt.tz_localize("UTC")

        # Average across zones if present
        temp_data = temp_data.groupby("time")[temp_varname].mean().reset_index()

        if read_stock_data:
            # 2018 subset to match NREL aggregate year in this workflow
            temp_data = temp_data[temp_data["time"].dt.year == 2018]

            # Read and stack all requested stock files (None filtered out by concat)
            parts = [
                read_savings(stock_type, building_type, upgrade)
                for building_type in building_types
                for upgrade in upgrades
            ]
            stock_data = pd.concat([p for p in parts if p is not None])

            # Merge on time
            df = pd.merge(stock_data, temp_data, on="time", how="inner")
        else:
            df = temp_data

        # Enrich with time features
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


def assign_loads_to_bus(
    stock_type,
    building_type,
    upgrade,
    climate_scenario,
):
    """
    Assign simulated savings to buses using county weights and spatial joins.
    """
    # Read simulated state-wide results
    sim_path = (
        PROJECT_PATH
        / "data" / "load" / stock_type / "simulated" / "state_wide"
        / f"{climate_scenario}_{upgrade}_{building_type}.csv"
    )
    df_simulated_load = pd.read_csv(sim_path)
    df_simulated_load["time"] = pd.to_datetime(df_simulated_load["time"])

    # Grid / geometry inputs
    gdf_bus = gpd.read_file(PROJECT_PATH / "data" / "grid" / "gis" / "Bus_clean.shp")
    df_npcc = pd.read_csv(PROJECT_PATH / "data" / "grid" / "npcc_new.csv")
    gdf_us = gpd.read_file(PROJECT_PATH / "data" / "nys" / "gis" / "cb_2018_us_county_5m.shp")
    gdf_nys = gdf_us[gdf_us["STATEFP"] == "36"]  # NY state only

    # NREL metadata restricted to NYISO
    df_meta = pd.read_parquet(PROJECT_PATH / "data" / "nrel" / stock_type / "baseline.parquet")
    df_meta = df_meta[df_meta["in.iso_rto_region"] == "NYISO"]
    df_meta["in.county_name"] = df_meta["in.county_name"].str.replace(" County", "")
    df_meta["in.county_name"] = df_meta["in.county_name"].str.replace("NY, ", "")

    # Building-type column in metadata
    if stock_type == "resstock":
        building_type_col = "in.geometry_building_type_recs"
    elif stock_type == "comstock":
        building_type_col = "in.comstock_building_type"
    else:
        raise ValueError("stock_type must be 'resstock' or 'comstock'")

    # County weights from metadata (normalized)
    df_meta_weights = (
        df_meta.groupby(["in.county_name", building_type_col])[["upgrade"]]
        .count()
        .reset_index()
        .rename(columns={"upgrade": "weight", "in.county_name": "county_name"})
    )
    df_meta_weights = df_meta_weights[
        df_meta_weights[building_type_col] == building_type_meta_map[building_type]
    ]
    df_meta_weights["weight"] = df_meta_weights["weight"] / df_meta_weights["weight"].sum()

    # Cross-join time series with county weights
    df_county_loads = pd.merge(
        df_simulated_load[["time", "predicted_savings_MW"]],
        df_meta_weights[["county_name", "weight"]],
        how="cross",
    )
    df_county_loads["county_load_MW"] = (
        df_county_loads["predicted_savings_MW"] * df_county_loads["weight"]
    )

    # Buses with nonzero load in NPCC, plus buses in zero-load zones (equal split)
    buses_with_load = df_npcc.query("sumLoadP0 > 0.")["busIdx"].to_numpy()
    buses_with_load = np.append(
        buses_with_load,
        df_npcc.set_index("zoneID")
        .loc[df_npcc.groupby("zoneID")["sumLoadP0"].sum() == 0.0]["busIdx"]
        .to_numpy(),
    )

    # County -> bus mapping for counties WITH a bus
    counties_with_bus = gpd.sjoin(
        gdf_bus[gdf_bus["bus_id"].isin(buses_with_load)][["bus_id", "geometry"]].to_crs(gdf_nys.crs),
        gdf_nys[["NAME", "geometry"]],
        how="inner",
        predicate="within",
    ).reset_index()
    counties_with_bus["county_to_bus_weight"] = counties_with_bus.groupby("NAME")["NAME"].transform(
        lambda x: 1.0 / x.count()
    )

    # County -> bus mapping for counties WITHOUT a bus (nearest)
    counties_with_no_bus_names = np.setdiff1d(gdf_nys["NAME"], counties_with_bus["NAME"])
    counties_with_no_bus = gpd.sjoin_nearest(
        gdf_nys[gdf_nys["NAME"].isin(counties_with_no_bus_names)][["NAME", "geometry"]],
        gdf_bus[gdf_bus["bus_id"].isin(buses_with_load)][["bus_id", "geometry"]].to_crs(gdf_nys.crs),
    ).reset_index()
    counties_with_no_bus["county_to_bus_weight"] = 1.0

    county_to_bus = pd.concat(
        [
            counties_with_bus[["NAME", "bus_id", "county_to_bus_weight"]],
            counties_with_no_bus[["NAME", "bus_id", "county_to_bus_weight"]],
        ],
        ignore_index=True,
    )

    # Join loads to bus mapping
    df_bus_loads = pd.merge(
        df_county_loads[["time", "county_name", "county_load_MW"]],
        county_to_bus,
        left_on="county_name",
        right_on="NAME",
        how="outer",
    )
    df_bus_loads["bus_load_MW"] = df_bus_loads["county_load_MW"] * df_bus_loads["county_to_bus_weight"]
    df_bus_loads = df_bus_loads.groupby(["time", "bus_id"])[["bus_load_MW"]].sum()
    return df_bus_loads
