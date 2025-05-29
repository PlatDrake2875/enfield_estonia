# xgboost_hackathon.py
# Energy Consumption Forecasting
# Section 6: Hackathon Simulation: Leave-One-Out Cross-Validation

# This script simulates the hackathon scenario:
# 1. Train on 9 buildings' full-year data.
# 2. Use 2 months of data from a "new" (held-out) building for fine-tuning/feature context.
# 3. Predict remaining 10 months for the new building.
# It performs leave-one-out cross-validation.

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split # For splitting the initial 2 months
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from datetime import timedelta # For DateOffset, already imported in notebook

CLEANED_XLSX = 'clean.xlsx'

# Cell 23: prepare_data function
def prepare_data(cleaned_xlsx_path):
    """Load and prepare data, ensuring unique datetime indices."""
    print("Preparing data for XGBoost Hackathon...")
    elec = pd.read_excel(cleaned_xlsx_path, sheet_name='Electricity kWh')
    weather = pd.read_excel(cleaned_xlsx_path, sheet_name='Weather archive')
    areas = pd.read_excel(cleaned_xlsx_path, sheet_name='Areas')
    
    elec['Datetime'] = pd.to_datetime(elec['time_iso'])
    weather['Datetime'] = pd.to_datetime(weather['time_iso'])
    
    elec = elec.groupby('Datetime').first().reset_index()
    weather = weather.groupby('Datetime').first().reset_index()
    
    elec = elec.set_index('Datetime').drop('time_iso', axis=1, errors='ignore')
    weather = weather.set_index('Datetime').drop('time_iso', axis=1, errors='ignore')
    
    weather = weather.resample('H').mean() # Resample weather to hourly
    
    common_idx = elec.index.intersection(weather.index)
    elec = elec.loc[common_idx]
    weather = weather.loc[common_idx]
    
    elec = elec.interpolate(method='time').ffill().bfill()
    weather = weather.interpolate(method='time').ffill().bfill()
    
    print("Data preparation complete.")
    return elec, weather, areas

# Cell 24: create_features function
def create_features(df_elec, df_weather, target_building, area_map_dict, lookback_hours=[1, 24, 168]):
    """Creates features for XGBoost model."""
    print(f"Creating features for {target_building}...")
    data_feat = pd.concat([df_elec[[target_building]], df_weather], axis=1)
    
    # Add lagged target variables
    for lag in lookback_hours:
        data_feat[f'{target_building}_lag_{lag}h'] = data_feat[target_building].shift(lag)
    
    # Add rolling means
    data_feat[f'{target_building}_rolling_24h_mean'] = data_feat[target_building].rolling(window=24, min_periods=1).mean().shift(1) # Shift mean to avoid data leakage

    # Add time-based features
    data_feat['hour'] = data_feat.index.hour
    data_feat['dayofweek'] = data_feat.index.dayofweek
    data_feat['month'] = data_feat.index.month
    data_feat['dayofyear'] = data_feat.index.dayofyear
    data_feat['weekofyear'] = data_feat.index.isocalendar().week.astype(int)


    # Add building size if available
    if target_building in area_map_dict:
        data_feat[f'{target_building}_size'] = area_map_dict[target_building]
    
    # Define feature columns (excluding the target itself)
    feature_columns = [col for col in data_feat.columns if col != target_building]
    
    data_feat = data_feat.dropna() # Drop rows with NaNs from lagging/rolling
    print(f"Features created for {target_building}. Shape: {data_feat.shape}")
    return data_feat, feature_columns

# Cell 25: hackathon_split function
def hackathon_split(data_with_features, target_building_col, train_months=2):
    """Splits data for the hackathon scenario."""
    print(f"Splitting data for {target_building_col} (hackathon setup)...")
    start_date = data_with_features.index.min()
    split_date = start_date + pd.DateOffset(months=train_months)
    
    train_df = data_with_features[data_with_features.index < split_date]
    test_df = data_with_features[data_with_features.index >= split_date]
    print("Data split complete.")
    return train_df, test_df

# Cell 26: cross_validate_buildings function (XGBoost LOOCV)
def cross_validate_buildings_xgboost(elec_df, weather_df, areas_df):
    """Performs leave-one-out cross-validation using XGBoost."""
    print("\nStarting XGBoost Leave-One-Out Cross-Validation...")
    buildings = [col for col in elec_df.columns if col != 'time_iso']
    area_map = areas_df.set_index('Buid_ID')['Area [m2]'].to_dict()
    results_cv = {}

    for held_out_building in buildings:
        print(f"\nProcessing with {held_out_building} as the held-out (new) building.")
        
        # Prepare training data from other buildings (full year)
        train_buildings = [b for b in buildings if b != held_out_building]
        
        X_train_list, y_train_list = [], []
        for train_bldg in train_buildings:
            bldg_elec = elec_df[[train_bldg]]
            # Pass full weather_df, features will be selected based on current bldg_elec
            data_bldg_feat, feature_cols_bldg = create_features(bldg_elec, weather_df.copy(), train_bldg, area_map)
            if not data_bldg_feat.empty:
                X_train_list.append(data_bldg_feat[feature_cols_bldg])
                y_train_list.append(data_bldg_feat[train_bldg])

        if not X_train_list:
            print(f"  No training data for other buildings when {held_out_building} is held out. Skipping.")
            continue
            
        X_train_all = pd.concat(X_train_list)
        y_train_all = pd.concat(y_train_list)

        # Prepare data for the held-out building
        held_out_elec = elec_df[[held_out_building]]
        data_held_out_feat, feature_cols_held_out = create_features(held_out_elec, weather_df.copy(), held_out_building, area_map)
        
        if data_held_out_feat.empty:
            print(f"  No data after feature creation for held-out building {held_out_building}. Skipping.")
            continue

        # Split held-out building data (2 months "train" context, 10 months "test")
        train_held_out, test_held_out = hackathon_split(data_held_out_feat, held_out_building)

        if train_held_out.empty or test_held_out.empty:
            print(f"  Not enough data to split for held-out building {held_out_building}. Skipping.")
            continue
            
        # Combine initial training data (other buildings) with 2 months of held-out building context
        # Ensure consistent feature sets; use intersection of columns
        common_features = list(set(X_train_all.columns) & set(train_held_out[feature_cols_held_out].columns))
        
        X_train_final = pd.concat([X_train_all[common_features], train_held_out[common_features]])
        y_train_final = pd.concat([y_train_all, train_held_out[held_out_building]])
        
        X_test_final = test_held_out[common_features]
        y_test_final = test_held_out[held_out_building]

        # Train/Val split for XGBoost internal validation (from the combined training set)
        X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
            X_train_final, y_train_final, test_size=0.2, random_state=42, shuffle=False # Time series, no shuffle
        )

        print(f"  Final Train shapes: X={X_train_xgb.shape}, y={y_train_xgb.shape}")
        print(f"  Final Val shapes: X={X_val_xgb.shape}, y={y_val_xgb.shape}")
        print(f"  Final Test shapes: X={X_test_final.shape}, y={y_test_final.shape}")

        dtrain_xgb = xgb.DMatrix(X_train_xgb, label=y_train_xgb)
        dval_xgb = xgb.DMatrix(X_val_xgb, label=y_val_xgb)
        dtest_xgb = xgb.DMatrix(X_test_final, label=y_test_final)
        
        params_xgb = {
            'tree_method': 'gpu_hist' if xgb.config.get_config().get('USE_CUDA', False) else 'hist', # Check GPU availability
            'predictor': 'gpu_predictor' if xgb.config.get_config().get('USE_CUDA', False) else 'cpu_predictor',
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'learning_rate': 0.01, # Slower LR
            'max_depth': 8,        # Adjusted depth
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'gamma': 0.05,
            'random_state': 42
        }

        print(f"  Training XGBoost for held-out building: {held_out_building}...")
        evals_result_xgb = {}
        model_xgb = xgb.train(
            params_xgb,
            dtrain_xgb,
            num_boost_round=500, # More rounds
            evals=[(dtrain_xgb, 'train'), (dval_xgb, 'eval')],
            early_stopping_rounds=50, # More patience
            verbose_eval=50,
            callbacks=[xgb.callback.record_evaluation(evals_result_xgb)]
        )
        
        predictions_xgb = model_xgb.predict(dtest_xgb)
        mape_xgb = mean_absolute_percentage_error(y_test_final, predictions_xgb) * 100
        
        results_cv[held_out_building] = {
            'mape': mape_xgb,
            'true': y_test_final.values,
            'pred': predictions_xgb,
            'dates': y_test_final.index,
            'model': model_xgb, # Store the trained model
            'evals_result': evals_result_xgb # Store training history
        }
        print(f"  MAPE for {held_out_building}: {mape_xgb:.2f}%")
        
    print("XGBoost LOOCV complete.")
    return results_cv

# Cell 27: plot_results function
def plot_xgboost_results(cv_results):
    """Plots MAPE comparison and actual vs. predicted for the best building."""
    print("\nPlotting XGBoost LOOCV Results...")
    # Plot 1: MAPE comparison
    plt.figure(figsize=(12, 6))
    mapes_cv = {k: v['mape'] for k, v in cv_results.items() if 'mape' in v} # Ensure 'mape' key exists
    
    if not mapes_cv:
        print("No MAPE results to plot.")
        return

    sorted_mapes = sorted(mapes_cv.items(), key=lambda item: item[1])
    buildings_sorted = [item[0] for item in sorted_mapes]
    mape_values_sorted = [item[1] for item in sorted_mapes]

    bars = plt.bar(buildings_sorted, mape_values_sorted)
    plt.xticks(rotation=45, ha='right')
    plt.title('XGBoost LOOCV: MAPE by Held-Out Building')
    plt.ylabel('MAPE (%)')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Actual vs Predicted for best performing building (lowest MAPE)
    if sorted_mapes:
        best_bldg_cv = sorted_mapes[0][0]
        plt.figure(figsize=(15, 6))
        plt.plot(cv_results[best_bldg_cv]['dates'], cv_results[best_bldg_cv]['true'], label='Actual', alpha=0.7)
        plt.plot(cv_results[best_bldg_cv]['dates'], cv_results[best_bldg_cv]['pred'], label='Predicted', alpha=0.7, linestyle='--')
        plt.title(f'XGBoost Predictions for {best_bldg_cv} (Best Performing in LOOCV)')
        plt.xlabel("Date")
        plt.ylabel("kWh")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No best building to plot predictions for.")


def main():
    """Main function to run XGBoost Hackathon simulation."""
    elec_data, weather_data, areas_data = prepare_data(CLEANED_XLSX)
    
    # Ensure elec_data is not empty before proceeding
    if elec_data.empty:
        print("Electricity data is empty after preparation. Exiting.")
        return

    cv_results_xgb = cross_validate_buildings_xgboost(elec_data, weather_data, areas_data)
    
    if cv_results_xgb: # Check if results were generated
        plot_xgboost_results(cv_results_xgb)
        # Print overall average MAPE
        all_mapes = [res['mape'] for res in cv_results_xgb.values() if 'mape' in res and not np.isnan(res['mape'])]
        if all_mapes:
            print(f"\nOverall Average MAPE from XGBoost LOOCV: {np.mean(all_mapes):.2f}%")
        else:
            print("No valid MAPE scores to average.")
    else:
        print("No cross-validation results to plot or average.")


if __name__ == '__main__':
    main()
