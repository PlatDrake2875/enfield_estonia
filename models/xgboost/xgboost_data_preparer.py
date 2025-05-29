# models/xgboost/xgboost_data_preparer.py

import pandas as pd
import numpy as np
import os
import re 

# Define the input data directory and filenames (output from data_cleaning.py)
INPUT_DATA_DIR = 'data' 
ELECTRICITY_CSV_FILENAME = 'cleaned_electricity.csv'
WEATHER_CSV_FILENAME = 'cleaned_weather.csv'
AREAS_CSV_FILENAME = 'cleaned_areas.csv'

def prepare_data():
    """Load and prepare data from cleaned CSV files."""
    print("Preparing data for XGBoost...")
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.dirname(current_script_dir) 
    project_root = os.path.dirname(models_dir) 
    
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
    weather_original_index = weather.set_index('Datetime').drop('time_iso', axis=1, errors='ignore')

    if 'phenomena_text' in weather_original_index.columns:
        weather_original_index['phenomena_text'] = weather_original_index['phenomena_text'].astype(str).fillna('')
    
    numeric_cols = weather_original_index.select_dtypes(include=np.number).columns.tolist()
    
    string_cols_for_resampling = []
    if 'phenomena_text' in weather_original_index.columns: 
        string_cols_for_resampling.append('phenomena_text')
    
    original_cloud_col_name = next((c for c in weather_original_index.columns if 'total cloud cover' in c.lower() or c.lower() == 'c'), None)
    if original_cloud_col_name and original_cloud_col_name not in numeric_cols:
         if original_cloud_col_name not in string_cols_for_resampling: 
            string_cols_for_resampling.append(original_cloud_col_name)

    weather_numeric_resampled = weather_original_index[numeric_cols].resample('h').mean()

    if string_cols_for_resampling:
        weather_strings_resampled = weather_original_index[string_cols_for_resampling].resample('h').ffill().bfill()
        weather_resampled = weather_numeric_resampled.join(weather_strings_resampled, how='left')
    else:
        weather_resampled = weather_numeric_resampled 
    
    weather = weather_resampled 
    weather.ffill(inplace=True) 
    weather.bfill(inplace=True)

    common_idx = elec.index.intersection(weather.index)
    elec = elec.loc[common_idx]
    weather = weather.loc[common_idx] 
    
    numeric_cols_elec = elec.select_dtypes(include=np.number).columns
    elec[numeric_cols_elec] = elec[numeric_cols_elec].interpolate(method='time').ffill().bfill()
    
    weather_numeric_cols_final = weather.select_dtypes(include=np.number).columns
    weather[weather_numeric_cols_final] = weather[weather_numeric_cols_final].interpolate(method='time').ffill().bfill()

    print("Data preparation complete.")
    return elec, weather, areas

def create_features(df_elec_single_building, df_weather_full, target_building_name, area_map_dict, lookback_hours=[1, 2, 3, 23, 24, 25, 47, 48, 49, 167, 168, 169]):
    """Creates features for XGBoost model for a single target building."""
    print(f"Creating features for {target_building_name}...")
    
    data_feat = df_elec_single_building[[target_building_name]].copy()
    data_feat = data_feat.join(df_weather_full, how='left') 

    if 'phenomena_text' in data_feat.columns:
        phenomena_text_series = data_feat['phenomena_text'].astype(str).fillna('').str.lower()
        
        phenomena_keywords = {
            'is_snow': ['snow', 'snowfall'], 'is_rain': ['rain', 'drizzle'],
            'is_fog': ['fog'], 'is_mist': ['mist'], 'is_shower': ['shower'],
            'is_drifting': ['drifting'], 'is_freezing_precip': ['freezing', 'supercooled'],
            'is_grains': ['grains'] 
        }
        
        for feature_name, keywords_list in phenomena_keywords.items():
            pattern = r'\b(?:' + '|'.join(keywords_list) + r')\b'
            data_feat[feature_name] = phenomena_text_series.str.contains(pattern, regex=True, case=False).astype(int)
        print(f"  Created binary features from 'phenomena_text' for {target_building_name}.")
    else:
        print(f"  Warning: 'phenomena_text' column not found in weather data for {target_building_name}. Skipping phenomena features.")

    for lag in lookback_hours:
        data_feat[f'{target_building_name}_lag_{lag}h'] = data_feat[target_building_name].shift(lag)
    
    windows = [3, 6, 12, 24, 48, 168] 
    for window in windows:
        data_feat[f'{target_building_name}_roll_mean_{window}h'] = data_feat[target_building_name].rolling(window=window, min_periods=1).mean().shift(1)
        data_feat[f'{target_building_name}_roll_std_{window}h'] = data_feat[target_building_name].rolling(window=window, min_periods=1).std().shift(1)

    # Time-based features
    data_feat['hour'] = data_feat.index.hour
    data_feat['dayofweek'] = data_feat.index.dayofweek 
    data_feat['dayofyear'] = data_feat.index.dayofyear
    data_feat['year'] = data_feat.index.year 
    # --- OMITTED FEATURES ---
    # data_feat['month'] = data_feat.index.month # Omitted
    # data_feat['weekofyear'] = data_feat.index.isocalendar().week.astype(int) # Omitted
    # data_feat['quarter'] = data_feat.index.quarter # Omitted
    # --- END OMITTED FEATURES ---

    data_feat['hour_x_dayofweek'] = data_feat['hour'] * data_feat['dayofweek']
    
    temp_col_found = None
    possible_temp_cols = ['Temperature', 'temp', 'air_temp', 'temperature', 'Air temperature', 'T', 'tt'] 
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
    
    cols_to_exclude_as_features = [target_building_name, 'phenomena_text'] 
    original_cloud_col_name_check = next((c for c in data_feat.columns if 'total cloud cover' in c.lower() or c.lower() == 'c'), None)
    if original_cloud_col_name_check:
        if original_cloud_col_name_check != 'cloud_cover_percentage':
            cols_to_exclude_as_features.append(original_cloud_col_name_check)
        
    feature_columns = [col for col in data_feat.columns if col not in cols_to_exclude_as_features]
    
    for col in feature_columns:
        if data_feat[col].dtype == 'object':
            print(f"  Warning: Feature column '{col}' is of object type. Attempting to convert to numeric.")
            data_feat[col] = pd.to_numeric(data_feat[col], errors='coerce')

    data_feat = data_feat.dropna(subset=[target_building_name]) 
    
    print(f"Features created for {target_building_name}. Shape: {data_feat.shape}")
    return data_feat, feature_columns

if __name__ == '__main__':
    # ... (if __name__ == '__main__' block remains the same) ...
    print("Testing xgboost_data_preparer.py (now in models/xgboost/)...")
    try:
        elec_df, weather_df, areas_df = prepare_data()
        print("\nElectricity data sample:\n", elec_df.head())
        print("\nWeather data sample (should include new features like 'cloud_cover_percentage', 'phenomena_text', 'atm_pressure_sea_level'):\n", weather_df.head())
        if 'phenomena_text' in weather_df.columns:
            print("\nSample of 'phenomena_text':\n", weather_df['phenomena_text'].dropna().unique()[:10]) 
        if 'cloud_cover_percentage' in weather_df.columns:
            print("\nSample of 'cloud_cover_percentage':\n", weather_df['cloud_cover_percentage'].value_counts(dropna=False))
        if 'atm_pressure_sea_level' in weather_df.columns: # This should no longer exist if dropped correctly
            print("\nSample of 'atm_pressure_sea_level':\n", weather_df['atm_pressure_sea_level'].describe())
        else:
            print("\n'atm_pressure_sea_level' column correctly omitted from weather_df.")


        print("\nAreas data sample:\n", areas_df.head())

        if not elec_df.empty:
            building_cols_test = [col for col in elec_df.columns if col not in ['time_iso', 'Datetime']]
            if building_cols_test:
                test_bldg = building_cols_test[0]
                area_map_test = areas_df.set_index('Buid_ID')['Area [m2]'].to_dict()
                
                features_df, feature_list = create_features(elec_df[[test_bldg]], weather_df, test_bldg, area_map_test)
                print(f"\nFeatures for {test_bldg} sample:\n", features_df.head())
                print(f"\nFeature list for {test_bldg} (first 10):", feature_list[:10])
                print(f"\nTotal features for {test_bldg}: {len(feature_list)}")
                omitted_date_features = ['month', 'quarter', 'weekofyear']
                for feat in omitted_date_features:
                    if feat in features_df.columns:
                        print(f"ERROR: Feature '{feat}' was NOT omitted.")
                    else:
                        print(f"Correct: Feature '{feat}' was omitted.")

                phenomena_related_features = [col for col in features_df.columns if col.startswith('is_')]
                if phenomena_related_features:
                    print(f"\nBinary phenomena features created for {test_bldg}:\n", features_df[phenomena_related_features].head())
                else:
                    print(f"\nNo binary phenomena features found for {test_bldg} in the sample output.")
            else:
                print("No building columns in electricity data to test feature creation.")
        else:
            print("Electricity data is empty, cannot test feature creation.")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
