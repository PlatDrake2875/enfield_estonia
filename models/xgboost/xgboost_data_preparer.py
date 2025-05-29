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

def create_features(df_elec_single_building, df_weather_full, target_building_name, area_map_dict, 
                    lookback_hours=None, 
                    rolling_windows=None
                    ):
    """Creates features for XGBoost model for a single target building."""
    print(f"Creating features for {target_building_name}...")

    if lookback_hours is None:
        lookback_hours = [1, 2, 3, 4, 5, 6, 12, 23, 24, 25, 47, 48, 49, 71, 72, 73, 167, 168, 169, 335, 336, 337]
    if rolling_windows is None:
        rolling_windows = [3, 6, 12, 24, 48, 168]
    
    data_feat = df_elec_single_building[[target_building_name]].copy()
    data_feat = data_feat.join(df_weather_full, how='left') 

    # --- 1. Phenomena Text Processing (Binary Flags) ---
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
        print(f"  Created binary features from 'phenomena_text'.")
    else:
        print(f"  Warning: 'phenomena_text' column not found. Skipping phenomena features.")

    # --- 2. Target Lagged Features ---
    print(f"  Creating target lags for {target_building_name}...")
    for lag in lookback_hours:
        data_feat[f'{target_building_name}_lag_{lag}h'] = data_feat[target_building_name].shift(lag)
    
    # --- 3. Target Rolling Window Features ---
    print(f"  Creating target rolling window features for {target_building_name}...")
    for window in rolling_windows:
        shifted_target = data_feat[target_building_name].shift(1) 
        data_feat[f'{target_building_name}_roll_mean_{window}h'] = shifted_target.rolling(window=window, min_periods=1).mean()
        data_feat[f'{target_building_name}_roll_std_{window}h'] = shifted_target.rolling(window=window, min_periods=1).std()
        data_feat[f'{target_building_name}_roll_min_{window}h'] = shifted_target.rolling(window=window, min_periods=1).min()
        data_feat[f'{target_building_name}_roll_max_{window}h'] = shifted_target.rolling(window=window, min_periods=1).max()
        data_feat[f'{target_building_name}_roll_median_{window}h'] = shifted_target.rolling(window=window, min_periods=1).median()

    # --- 4. Time-Based Features (Raw, Cyclical, Weekend, Season) ---
    print(f"  Creating time-based features...")
    idx = data_feat.index
    data_feat['hour'] = idx.hour
    data_feat['dayofweek'] = idx.dayofweek # Monday=0, Sunday=6
    data_feat['dayofyear'] = idx.dayofyear
    data_feat['year'] = idx.year 
    
    # Cyclical features
    data_feat['hour_sin'] = np.sin(2 * np.pi * data_feat['hour'] / 24.0)
    data_feat['hour_cos'] = np.cos(2 * np.pi * data_feat['hour'] / 24.0)
    data_feat['dayofweek_sin'] = np.sin(2 * np.pi * data_feat['dayofweek'] / 7.0)
    data_feat['dayofweek_cos'] = np.cos(2 * np.pi * data_feat['dayofweek'] / 7.0)
    data_feat['dayofyear_sin'] = np.sin(2 * np.pi * data_feat['dayofyear'] / 365.25)
    data_feat['dayofyear_cos'] = np.cos(2 * np.pi * data_feat['dayofyear'] / 365.25)
    
    # is_weekend feature
    data_feat['is_weekend'] = data_feat['dayofweek'].apply(lambda x: 1 if x >= 5 else 0) # 5=Saturday, 6=Sunday
    print(f"  Created 'is_weekend' feature.")

    # Seasonality feature (categorical then one-hot encoded)
    month_col = idx.month # Get month from index
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Autumn' # 9, 10, 11
    
    data_feat['season_cat'] = month_col.to_series(index=data_feat.index).apply(get_season) # Ensure Series has same index
    
    # One-hot encode 'season_cat'
    season_dummies = pd.get_dummies(data_feat['season_cat'], prefix='season', dtype=int)
    data_feat = pd.concat([data_feat, season_dummies], axis=1)
    # data_feat.drop('season_cat', axis=1, inplace=True) # Drop original categorical season column
    print(f"  Created one-hot encoded season features: {list(season_dummies.columns)}")
    
    # --- 5. Weather Features (Direct, Lags, Rolling) ---
    print(f"  Creating weather-based features...")
    temp_col_found = None
    possible_temp_cols = ['Temperature', 'temp', 'air_temp', 'temperature', 'Air temperature', 'T', 'tt'] 
    for col_name_df in data_feat.columns: 
        for p_name in possible_temp_cols:
            if col_name_df.strip().lower() == p_name.strip().lower():
                temp_col_found = col_name_df
                break
        if temp_col_found:
            break
            
    weather_features_to_roll = []
    if temp_col_found:
        weather_features_to_roll.append(temp_col_found)
        print(f"    Identified temperature column: '{temp_col_found}'")
    else:
        print(f"    Warning: No standard temperature column found for rolling features.")
    
    if 'cloud_cover_percentage' in data_feat.columns:
        weather_features_to_roll.append('cloud_cover_percentage')
    if 'visibility' in data_feat.columns: 
        weather_features_to_roll.append('visibility')
    rh_col_found = next((col for col in data_feat.columns if 'relative humidity' in col.lower()), None)
    if rh_col_found:
        weather_features_to_roll.append(rh_col_found)
        print(f"    Identified humidity column: '{rh_col_found}'")

    weather_rolling_windows = [3, 6, 12, 24] 
    weather_lags = [1, 2, 3, 24]

    for w_feat in weather_features_to_roll:
        if w_feat in data_feat.columns:
            for lag in weather_lags:
                data_feat[f'{w_feat}_lag_{lag}h'] = data_feat[w_feat].shift(lag)
            shifted_w_feat = data_feat[w_feat].shift(1)
            for window in weather_rolling_windows:
                data_feat[f'{w_feat}_roll_mean_{window}h'] = shifted_w_feat.rolling(window=window, min_periods=1).mean()
                data_feat[f'{w_feat}_roll_std_{window}h'] = shifted_w_feat.rolling(window=window, min_periods=1).std()
        else:
            print(f"    Warning: Column '{w_feat}' not found for weather rolling features/lags.")

    # --- 6. Interaction Features ---
    data_feat['hour_x_dayofweek'] = data_feat['hour'] * data_feat['dayofweek'] 
    if temp_col_found:
         data_feat['temp_x_hour'] = data_feat[temp_col_found] * data_feat['hour']
         print(f"    Created '{temp_col_found}_x_hour' interaction feature.")

    # --- 7. Building Size ---
    if target_building_name in area_map_dict:
        data_feat[f'building_size_m2'] = area_map_dict[target_building_name]
    else:
        data_feat[f'building_size_m2'] = np.nan 
    
    # --- Finalize Feature List & Clean up ---
    cols_to_exclude_as_features = [target_building_name, 'phenomena_text', 'season_cat'] # Add 'season_cat'
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
    
    print(f"Features created for {target_building_name}. Final shape: {data_feat.shape}, Number of features: {len(feature_columns)}")
    return data_feat, feature_columns

if __name__ == '__main__':
    print("Testing xgboost_data_preparer.py (now in models/xgboost/)...")
    try:
        elec_df, weather_df, areas_df = prepare_data()
        print("\nElectricity data sample:\n", elec_df.head())
        print("\nWeather data sample:\n", weather_df.head())
        
        if not elec_df.empty and not weather_df.empty:
            building_cols_test = [col for col in elec_df.columns if col not in ['time_iso', 'Datetime']]
            if building_cols_test:
                test_bldg = building_cols_test[0]
                area_map_test = areas_df.set_index('Buid_ID')['Area [m2]'].to_dict()
                
                features_df, feature_list = create_features(elec_df[[test_bldg]], weather_df.copy(), test_bldg, area_map_test)
                print(f"\nFeatures for {test_bldg} sample (first 5 rows, 10 cols):\n", features_df.iloc[:5, :10])
                print(f"\nFeature list for {test_bldg} (first 10):", feature_list[:10])
                print(f"\nTotal features for {test_bldg}: {len(feature_list)}")
                
                print("\nChecking for new feature types:")
                newly_added_features_to_check = ['is_weekend', 'season_Winter', 'season_Spring', 'season_Summer', 'season_Autumn']
                for new_feat in newly_added_features_to_check:
                    if new_feat in feature_list:
                        print(f"  Found new feature: {new_feat}")
                    else:
                         print(f"  MISSING expected new feature: {new_feat}")

                omitted_date_features = ['month', 'quarter', 'weekofyear'] # These should still be omitted
                for feat in omitted_date_features:
                    if feat in features_df.columns or feat in feature_list :
                        print(f"  ERROR: Feature '{feat}' was NOT omitted.")
                    else:
                        print(f"  Correct: Feature '{feat}' was omitted.")
            else:
                print("No building columns in electricity data to test feature creation.")
        else:
            print("Electricity or weather data is empty, cannot test feature creation.")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
