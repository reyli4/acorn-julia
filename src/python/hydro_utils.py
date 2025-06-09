import pandas as pd
import numpy as np


def disaggregate_weekly_to_hourly(
    df, method="average", morning_peak_hour=8, evening_peak_hour=18
):
    """
    Disaggregate weekly hydropower data to hourly data.

    Parameters:
    -----------
    df : pandas.DataFrame
        Weekly data with columns: datetime, eia_id, power_predicted_mwh, p_max, p_min, p_avg
    method : str
        'average' for constant hourly values or 'diurnal' for daily cycle
    morning_peak_hour : int
        Hour of day for morning peak (0-23), used only for diurnal method
    evening_peak_hour : int
        Hour of day for evening peak (0-23), used only for diurnal method

    Returns:
    --------
    pandas.DataFrame
        Hourly data with datetime and power_mw columns, plus original metadata
    """

    def create_diurnal_pattern(morning_peak=8, evening_peak=18):
        """Create a 24-hour normalized diurnal pattern with two peaks."""
        hours = np.arange(24)

        # Create two gaussian-like peaks
        morning_component = np.exp(-0.5 * ((hours - morning_peak) / 2.5) ** 2)
        evening_component = np.exp(-0.5 * ((hours - evening_peak) / 3.0) ** 2)

        # Combine peaks with evening peak being stronger
        pattern = 0.6 * morning_component + 1.0 * evening_component

        # Add baseline to avoid zeros
        pattern += 0.3

        # Normalize so mean = 1 (preserves total energy)
        pattern = pattern / pattern.mean()

        return pattern

    def create_pattern_for_period(daily_pattern, n_hours):
        """Create pattern for given number of hours."""
        n_full_days = n_hours // 24
        remaining_hours = n_hours % 24

        # Create pattern for full days
        if n_full_days > 0:
            full_days_pattern = np.tile(daily_pattern, n_full_days)
        else:
            full_days_pattern = np.array([])

        # Add partial day if needed
        if remaining_hours > 0:
            partial_day_pattern = daily_pattern[:remaining_hours]
            pattern = np.concatenate([full_days_pattern, partial_day_pattern])
        else:
            pattern = full_days_pattern

        return pattern

    def scale_pattern_to_constraints(pattern, p_avg, p_max, p_min, total_mwh):
        """Scale pattern to match average, and respect max/min constraints."""
        # Start with pattern scaled to match average
        scaled_pattern = pattern * p_avg

        # Check if we violate constraints
        pattern_max = scaled_pattern.max()
        pattern_min = scaled_pattern.min()

        # If we exceed p_max, compress the pattern
        if pattern_max > p_max:
            # Compress pattern to fit within [p_min, p_max]
            pattern_range = pattern.max() - pattern.min()
            available_range = p_max - p_min

            if pattern_range > 0:
                compression_factor = available_range / pattern_range
                scaled_pattern = p_min + (pattern - pattern.min()) * compression_factor

        # Final adjustment to ensure total energy conservation
        current_total = scaled_pattern.sum()
        target_total = total_mwh
        adjustment_factor = target_total / current_total
        scaled_pattern *= adjustment_factor

        return scaled_pattern

    if method == "average":
        # Average and ffill
        final_df = (
            df.set_index("datetime")
            .groupby("eia_id")
            .resample("h")[["p_avg"]]
            .ffill()
            .reset_index()
        )

    else:  # diurnal method
        # Process each plant separately for diurnal method (needs the complex logic)
        result_dfs = []

        for eia_id in df["eia_id"].unique():
            plant_df = df[df["eia_id"] == eia_id].copy()

            hourly_data = []

            for _, row in plant_df.iterrows():
                period_start = pd.to_datetime(row["datetime"])
                n_hours = int(row["n_hours"])

                # Create diurnal cycle
                daily_pattern = create_diurnal_pattern(
                    morning_peak_hour, evening_peak_hour
                )
                period_pattern = create_pattern_for_period(daily_pattern, n_hours)

                # Scale pattern to match constraints and conserve energy
                hourly_power = scale_pattern_to_constraints(
                    period_pattern,
                    row["p_avg"],
                    row["p_max"],
                    row["p_min"],
                    row["power_predicted_mwh"],
                )

                # Create hourly timestamps
                hourly_timestamps = pd.date_range(
                    start=period_start, periods=n_hours, freq="H"
                )

                # Create hourly records
                for i, (timestamp, power_mw) in enumerate(
                    zip(hourly_timestamps, hourly_power)
                ):
                    hourly_data.append(
                        {
                            "datetime": timestamp,
                            "eia_id": row["eia_id"],
                            "power_mw": power_mw,
                            # "period_start": period_start,
                            # "hour_of_period": i,
                            # "hour_of_day": timestamp.hour,
                            # "day_of_week": timestamp.dayofweek,
                            # "scenario": row["scenario"],
                            # "original_n_hours": n_hours,
                            # "original_p_avg": row["p_avg"],
                            # "original_p_max": row["p_max"],
                            # "original_p_min": row["p_min"],
                            # "original_total_mwh": row["power_predicted_mwh"],
                        }
                    )

            plant_hourly_df = pd.DataFrame(hourly_data)
            result_dfs.append(plant_hourly_df)

        # Combine all plants
        final_df = pd.concat(result_dfs, ignore_index=True)

    # Sort by plant and datetime
    final_df = final_df.set_index(["eia_id", "datetime"])

    return final_df
