# baseline_models.py
# Energy Consumption Forecasting
# Section 4: Baseline Models

# This script implements three simple forecasting approaches:
# 1. Persistence: y_hat_t = y_{t-24}
# 2. Seasonal ARIMA: SARIMA(1,0,1)x(1,1,1,24)
# 3. Holt-Winters: Additive exponential smoothing, period = 24h

import pandas as pd
import numpy as np
import time
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

CLEANED_XLSX = 'clean.xlsx'

# Cell 10: Imports & Helpers
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, ignoring zeros."""
    mask = y_true != 0
    # Handle cases where y_true[mask] might be empty or all zeros
    if np.sum(mask) == 0 or np.all(y_true[mask] == 0):
        return np.nan # Or some other indicator of an issue
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def load_and_split_data(cleaned_xlsx_path):
    """Loads cleaned electricity data and splits into train/test."""
    print("Loading and splitting data for baseline models...")
    elec = pd.read_excel(cleaned_xlsx_path, sheet_name='Electricity kWh')
    elec['Datetime'] = pd.to_datetime(elec['time_iso'])
    elec.set_index('Datetime', inplace=True)
    
    start_date = elec.index.min()
    train_end_date = start_date + pd.DateOffset(months=2)
    
    # Ensure unique indices for splitting
    elec_unique_idx = elec[~elec.index.duplicated(keep='first')]

    train = elec_unique_idx[:train_end_date - pd.Timedelta(seconds=1)]
    test  = elec_unique_idx[train_end_date:]
    print("Data loaded and split.")
    return train, test, elec_unique_idx # Return elec_unique_idx for persistence

def run_persistence_baseline(train_df_unique, test_df_unique):
    """Runs the persistence (naive) baseline model."""
    # Cell 11: Persistence (naive) baseline
    print("Running persistence baseline...")
    persist_mape_results = {}
    
    # Iterate over building columns (excluding 'time_iso')
    building_cols = [col for col in test_df_unique.columns if col != 'time_iso']

    for bld in building_cols:
        y_true = test_df_unique[bld].values
        # Forecast: shift by 24h from the unique-indexed training data
        # Reindex to test_df_unique.index to ensure alignment and proper length
        y_pred_series = train_df_unique[bld].shift(24).reindex(test_df_unique.index)
        y_pred_series.ffill(inplace=True) # Forward fill NaNs from shift and reindex
        y_pred_series.bfill(inplace=True) # Backward fill any remaining NaNs at the beginning
        y_pred = y_pred_series.values
        
        persist_mape_results[bld] = mape(y_true, y_pred)
    print("Persistence baseline complete.")
    return persist_mape_results

def run_sarima_baseline(train_df, test_df):
    """Runs the SARIMA baseline model."""
    # Cell 12: SARIMA
    print("Running SARIMA baseline...")
    warnings.filterwarnings("ignore") # Suppress SARIMAX warnings
    sarima_mape_results = {}
    t0 = time.time()

    building_cols = [col for col in test_df.columns if col != 'time_iso']

    for bld in building_cols:
        print(f"  Fitting SARIMA for building: {bld}")
        # Fit SARIMA on a single building's data
        # Ensure train_df[bld] has a DatetimeIndex with a defined frequency
        train_series = train_df[bld].asfreq('H') # Assuming hourly data, adjust if different
        
        # Handle potential NaNs from asfreq if original data had gaps
        train_series.interpolate(method='time', inplace=True)
        train_series.bfill(inplace=True) # Fill any remaining at the start
        train_series.ffill(inplace=True) # Fill any remaining at the end
        
        if train_series.isnull().any():
            print(f"    Warning: NaNs still present in training series for {bld} after interpolation. Skipping SARIMA for this building.")
            sarima_mape_results[bld] = np.nan
            continue

        try:
            mod = SARIMAX(train_series, order=(1,0,1), 
                          seasonal_order=(1,1,1,24), 
                          enforce_stationarity=False,
                          enforce_invertibility=False) # Added to aid convergence
            res = mod.fit(disp=False, maxiter=50) # Reduced maxiter for speed, adjust as needed
            
            # Forecast for the length of the test set
            forecast_start = test_df.index.min()
            forecast_end = test_df.index.max()
            f = res.predict(start=forecast_start, end=forecast_end)
            
            # Ensure forecast aligns with test_df index
            f = f.reindex(test_df.index).ffill().bfill()

            sarima_mape_results[bld] = mape(test_df[bld].values, f.values)
        except Exception as e:
            print(f"    Error fitting SARIMA for {bld}: {e}")
            sarima_mape_results[bld] = np.nan

    print(f"SARIMA baseline done in {time.time()-t0:.1f}s")
    warnings.filterwarnings("default") # Restore warnings
    return sarima_mape_results

def run_holt_winters_baseline(train_df, test_df):
    """Runs the Holt-Winters baseline model."""
    # Cell 13: Holt-Winters
    print("Running Holt-Winters baseline...")
    hw_mape_results = {}
    t0 = time.time()

    building_cols = [col for col in test_df.columns if col != 'time_iso']

    for bld in building_cols:
        print(f"  Fitting Holt-Winters for building: {bld}")
        # Ensure train_df[bld] has a DatetimeIndex with a defined frequency
        train_series = train_df[bld].asfreq('H')
        train_series.interpolate(method='time', inplace=True)
        train_series.bfill(inplace=True)
        train_series.ffill(inplace=True)

        if train_series.isnull().any() or len(train_series) < 2 * 24: # HW needs at least 2 full seasons
            print(f"    Warning: Insufficient data or NaNs for Holt-Winters on {bld}. Skipping.")
            hw_mape_results[bld] = np.nan
            continue
        
        try:
            mod = ExponentialSmoothing(train_series, trend='add', seasonal='add', seasonal_periods=24)
            res = mod.fit(optimized=True)
            f = res.forecast(len(test_df))
            
            # Ensure forecast aligns with test_df index
            f_aligned = pd.Series(f.values, index=test_df.index).reindex(test_df.index).ffill().bfill()

            hw_mape_results[bld] = mape(test_df[bld].values, f_aligned.values)
        except Exception as e:
            print(f"    Error fitting Holt-Winters for {bld}: {e}")
            hw_mape_results[bld] = np.nan
            
    print(f"Holt-Winters baseline done in {time.time()-t0:.1f}s")
    return hw_mape_results

def display_baseline_results(persist_mapes, sarima_mapes, hw_mapes):
    """Displays the baseline model results."""
    # Cell 14: Results DataFrame
    print("\nBaseline Model MAPE Results (%):")
    baseline_df = pd.DataFrame({
        'Persistence': persist_mapes,
        'SARIMA': sarima_mapes,
        'HoltWinters': hw_mapes
    })
    # Calculate Mean_MAPE ensuring only numeric columns are used and NaNs are handled
    numeric_cols = baseline_df.select_dtypes(include=np.number).columns
    baseline_df['Mean_MAPE'] = baseline_df[numeric_cols].mean(axis=1, skipna=True)
    
    # Format for display
    styled_df = baseline_df.sort_values('Mean_MAPE').style.format("{:.2f}%", na_rep="N/A")
    print(styled_df.to_string()) # Print styled DataFrame as string for .py output

def main():
    """Main function to run baseline models."""
    train, test, elec_unique_idx = load_and_split_data(CLEANED_XLSX)
    
    # For persistence, use unique-indexed train and test sets
    # Ensure test_unique is created from the same base as test
    test_start_date = test.index.min()
    test_unique = elec_unique_idx[elec_unique_idx.index >= test_start_date]
    
    persist_results = run_persistence_baseline(elec_unique_idx, test_unique) # Pass unique-indexed train for shift
    sarima_results = run_sarima_baseline(train, test)
    hw_results = run_holt_winters_baseline(train, test)
    display_baseline_results(persist_results, sarima_results, hw_results)

if __name__ == '__main__':
    main()
