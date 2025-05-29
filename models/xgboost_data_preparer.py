# models/xgboost_data_preparer.py

import pandas as pd
import numpy as np
import os

# Define the input data directory and filenames (output from data_cleaning.py)
INPUT_DATA_DIR = 'data' # Assuming 'data' folder is at the same level as 'models' folder
ELECTRICITY_CSV_FILENAME = 'cleaned_electricity.csv'
WEATHER_CSV_FILENAME = 'cleaned_weather.csv'
AREAS_CSV_FILENAME = 'cleaned_areas.csv'

def prepare_data():
    """Load and prepare data from cleaned CSV files."""
    print("Preparing data for XGBoost...")
    
    # Construct full paths relative to the script's location or a defined project root
    # Assuming this script is in 'models' and 'data' is a sibling directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir) # Moves up one level from 'models' to project root
    
    data_dir_path = os.path.join(project_root, INPUT_DATA_DIR)

    electricity_path = os.path.join(data_dir_path, ELECTRICITY_CSV_FILENAME)
    weather_path = os.path.join(data_dir_path, WEATHER_CSV_FILENAME)
    areas_path = os.path.join(data_dir_path, AREAS_CSV_FILENAME)

    try:
        elec = pd.read_csv(electricity_path)
        weather = pd.read_csv(weather_path)
        areas = pd.read_csv(areas_path)
    except FileNotFoundError as e:
        print(f"Error: One or more cleaned CSV files not found in '{data_dir_path}'. Please run data_cleaning.py first.")
        print(f"Details: {e}")
        raise 

    elec['Datetime'] = pd.to_datetime(elec['time_iso'])
    weather['Datetime'] = pd.to_datetime(weather['time_iso'])
    
    elec = elec.loc[~elec['Datetime'].duplicated(keep='first')]
    weather = weather.loc[~weather['Datetime'].duplicated(keep='first')]
    
    elec = elec.set_index('Datetime').drop('time_iso', axis=1, errors='ignore')
    weather = weather.set_index('Datetime').drop('time_iso', axis=1, errors='ignore')
    
    numeric_weather_cols = weather.select_dtypes(include=np.number).columns
    weather_numeric_resampled = weather[numeric_weather_cols].resample('h').mean() 
    weather = weather_numeric_resampled
    
    common_idx = elec.index.intersection(weather.index)
    elec = elec.loc[common_idx]
    weather = weather.loc[common_idx]
    
    elec = elec.interpolate(method='time').ffill().bfill()
    weather = weather.interpolate(method='time').ffill().bfill()
    
    print("Data preparation complete.")
    return elec, weather, areas

def create_features(df_elec_single_building, df_weather_full, target_building_name, area_map_dict, lookback_hours=[1, 2, 3, 23, 24, 25, 47, 48, 49, 167, 168, 169]):
    """Creates features for XGBoost model for a single target building."""
    print(f"Creating features for {target_building_name}...")
    
    data_feat = df_elec_single_building[[target_building_name]].copy()
    data_feat = data_feat.join(df_weather_full, how='left') 

    for lag in lookback_hours:
        data_feat[f'{target_building_name}_lag_{lag}h'] = data_feat[target_building_name].shift(lag)
    
    windows = [3, 6, 12, 24, 48, 168] 
    for window in windows:
        data_feat[f'{target_building_name}_roll_mean_{window}h'] = data_feat[target_building_name].rolling(window=window, min_periods=1).mean().shift(1)
        data_feat[f'{target_building_name}_roll_std_{window}h'] = data_feat[target_building_name].rolling(window=window, min_periods=1).std().shift(1)

    data_feat['hour'] = data_feat.index.hour
    data_feat['dayofweek'] = data_feat.index.dayofweek 
    data_feat['dayofyear'] = data_feat.index.dayofyear
    data_feat['month'] = data_feat.index.month
    data_feat['weekofyear'] = data_feat.index.isocalendar().week.astype(int)
    data_feat['quarter'] = data_feat.index.quarter
    data_feat['year'] = data_feat.index.year 

    data_feat['hour_x_dayofweek'] = data_feat['hour'] * data_feat['dayofweek']
    
    temp_col_found = None
    possible_temp_cols = ['Temperature', 'temp', 'air_temp', 'temperature', 'Air temperature', 'T'] 
    for col_name in possible_temp_cols:
        matching_cols = [col for col in data_feat.columns if col.strip().lower() == col_name.strip().lower()]
        if matching_cols:
            temp_col_found = matching_cols[0] 
            break
            
    if temp_col_found:
         data_feat['temp_x_hour'] = data_feat[temp_col_found] * data_feat['hour']
         print(f"  Used '{temp_col_found}' for temperature interaction feature in {target_building_name}.")
    else:
        print(f"  Warning: No standard temperature column found for interaction feature in {target_building_name}.")

    if target_building_name in area_map_dict:
        data_feat[f'building_size_m2'] = area_map_dict[target_building_name]
    else:
        data_feat[f'building_size_m2'] = np.nan 
    
    feature_columns = [col for col in data_feat.columns if col != target_building_name]
    data_feat = data_feat.dropna(subset=[target_building_name]) 
    
    print(f"Features created for {target_building_name}. Shape: {data_feat.shape}")
    return data_feat, feature_columns

if __name__ == '__main__':
    # Example usage (optional, for testing this module directly)
    print("Testing xgboost_data_preparer.py...")
    try:
        elec_df, weather_df, areas_df = prepare_data()
        print("Electricity data sample:\n", elec_df.head())
        print("Weather data sample:\n", weather_df.head())
        print("Areas data sample:\n", areas_df.head())

        if not elec_df.empty:
            building_cols_test = [col for col in elec_df.columns if col not in ['time_iso', 'Datetime']]
            if building_cols_test:
                test_bldg = building_cols_test[0]
                area_map_test = areas_df.set_index('Buid_ID')['Area [m2]'].to_dict()
                features_df, feature_list = create_features(elec_df[[test_bldg]], weather_df, test_bldg, area_map_test)
                print(f"Features for {test_bldg} sample:\n", features_df.head())
                print(f"Feature list for {test_bldg}:", feature_list[:5]) # Print first 5 features
            else:
                print("No building columns in electricity data to test feature creation.")
        else:
            print("Electricity data is empty, cannot test feature creation.")

    except Exception as e:
        print(f"Error during testing: {e}")
