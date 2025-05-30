# models/xgboost/xgboost_data_preparer.py

import pandas as pd
import numpy as np
import os
import re 
import json 

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
                    output_paths, 
                    lookback_hours=None, 
                    rolling_windows=None,
                    force_recreate_features=False 
                    ):
    """Creates features for XGBoost model for a single target building, with caching."""
    
    feature_cache_dir = output_paths.get('feature_cache_dir')
    sanitized_building_name = "".join(c if c.isalnum() else "_" for c in target_building_name)
    cache_file_df = None
    cache_file_cols = None

    if feature_cache_dir:
        os.makedirs(feature_cache_dir, exist_ok=True)
        cache_file_df = os.path.join(feature_cache_dir, f"features_{sanitized_building_name}.parquet")
        cache_file_cols = os.path.join(feature_cache_dir, f"feature_list_{sanitized_building_name}.json")

        if not force_recreate_features and os.path.exists(cache_file_df) and os.path.exists(cache_file_cols):
            try:
                print(f"Loading cached features for {target_building_name} from {cache_file_df}")
                data_feat_cached = pd.read_parquet(cache_file_df)
                with open(cache_file_cols, 'r') as f:
                    feature_columns_cached = json.load(f)
                
                if not isinstance(data_feat_cached.index, pd.DatetimeIndex) and 'Datetime' in data_feat_cached.columns:
                    data_feat_cached['Datetime'] = pd.to_datetime(data_feat_cached['Datetime'])
                    data_feat_cached = data_feat_cached.set_index('Datetime') 
                elif not isinstance(data_feat_cached.index, pd.DatetimeIndex):
                     print("Warning: Cached feature DataFrame index is not DatetimeIndex and 'Datetime' column not found for re-indexing.")

                if target_building_name in data_feat_cached.columns and \
                   all(col in data_feat_cached.columns for col in feature_columns_cached[:min(5, len(feature_columns_cached))]):
                    print(f"Successfully loaded cached features for {target_building_name}.")
                    return data_feat_cached, feature_columns_cached
                else:
                    print(f"Warning: Cached features for {target_building_name} seem incomplete or corrupted. Recreating.")
            except Exception as e:
                print(f"Warning: Could not load cached features for {target_building_name} due to error: {e}. Recreating.")

    print(f"Creating features for {target_building_name} (cache miss or forced recreate)...")

    if lookback_hours is None:
        lookback_hours = [1, 2, 3, 4, 5, 6, 12, 23, 24, 25, 47, 48, 49, 71, 72, 73, 167, 168, 169, 335, 336, 337]
    if rolling_windows is None:
        rolling_windows = [3, 6, 12, 24, 48, 168]
    
    data_feat = df_elec_single_building[[target_building_name]].copy()
    data_feat = data_feat.join(df_weather_full, how='left') 
    
    # List to collect new feature Series
    new_features_to_concat = [] 

    # --- 0. Add Building Size and Derived Area Features ---
    building_size_col = 'building_size_m2' 
    log_building_size_col = 'log_building_size_m2'
    
    current_building_area = area_map_dict.get(target_building_name, np.nan)
    # Add directly to data_feat as they are needed early
    data_feat[building_size_col] = pd.to_numeric(current_building_area, errors='coerce') 
    data_feat[building_size_col].fillna(0, inplace=True)
    data_feat[log_building_size_col] = np.log1p(data_feat[building_size_col])
    print(f"  Added '{building_size_col}' and '{log_building_size_col}' features.")

    all_areas = pd.Series(area_map_dict.values()).dropna() 
    area_bin_cat_temp_col = 'area_bin_cat_temp' 
    data_feat[area_bin_cat_temp_col] = -1 # Initialize temp column in data_feat

    if not all_areas.empty and all_areas.nunique() > 1 : 
        try:
            num_quantiles = min(4, all_areas.nunique()) 
            if num_quantiles > 1:
                _, bin_edges = pd.qcut(all_areas, q=num_quantiles, labels=False, retbins=True, duplicates='drop')
                bin_edges = np.unique(bin_edges) 
                if len(bin_edges) > 1:
                    data_feat[area_bin_cat_temp_col] = pd.cut(data_feat[building_size_col], bins=bin_edges, labels=False, include_lowest=True, duplicates='drop')
                    data_feat[area_bin_cat_temp_col].fillna(-1, inplace=True) 
                    
                    area_bin_dummies = pd.get_dummies(data_feat[area_bin_cat_temp_col], prefix='area_qbin', dtype=int)
                    for col in area_bin_dummies.columns: 
                        new_features_to_concat.append(area_bin_dummies[col].rename(col)) # Collect dummies
                    print(f"  Created one-hot encoded area quantile bins: {list(area_bin_dummies.columns)}")
        except Exception as e_bin:
            print(f"  Warning: Could not create quantile bins for area: {e_bin}")
            # data_feat[area_bin_cat_temp_col] is already initialized

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
            new_features_to_concat.append(phenomena_text_series.str.contains(pattern, regex=True, case=False).astype(int).rename(feature_name))
        print(f"  Created binary features from 'phenomena_text'.")

    # --- 2. Target Lagged Features ---
    print(f"  Creating target lags for {target_building_name}...")
    for lag in lookback_hours:
        new_features_to_concat.append(data_feat[target_building_name].shift(lag).rename(f'{target_building_name}_lag_{lag}h'))
    
    # --- 3. Target Rolling Window Features ---
    print(f"  Creating target rolling window features for {target_building_name}...")
    for window in rolling_windows:
        shifted_target = data_feat[target_building_name].shift(1) 
        new_features_to_concat.append(shifted_target.rolling(window=window, min_periods=1).agg("mean").rename(f'{target_building_name}_roll_mean_{window}h'))
        new_features_to_concat.append(shifted_target.rolling(window=window, min_periods=1).agg("std").rename(f'{target_building_name}_roll_std_{window}h'))
        new_features_to_concat.append(shifted_target.rolling(window=window, min_periods=1).agg("min").rename(f'{target_building_name}_roll_min_{window}h'))
        new_features_to_concat.append(shifted_target.rolling(window=window, min_periods=1).agg("max").rename(f'{target_building_name}_roll_max_{window}h'))
        new_features_to_concat.append(shifted_target.rolling(window=window, min_periods=1).agg("median").rename(f'{target_building_name}_roll_median_{window}h'))

    # --- 4. Time-Based Features (Raw, Cyclical, Weekend, Season) ---
    print(f"  Creating time-based features...")
    idx = data_feat.index
    hour_s = pd.Series(idx.hour, index=idx, name='hour') # Explicit Series
    dayofweek_s = pd.Series(idx.dayofweek, index=idx, name='dayofweek') # Explicit Series
    dayofyear_s = pd.Series(idx.dayofyear, index=idx, name='dayofyear') # Explicit Series
    
    new_features_to_concat.extend([hour_s, dayofweek_s, dayofyear_s, pd.Series(idx.year, index=idx, name='year')])
    
    new_features_to_concat.append(np.sin(2 * np.pi * hour_s / 24.0).rename('hour_sin'))
    new_features_to_concat.append(np.cos(2 * np.pi * hour_s / 24.0).rename('hour_cos'))
    new_features_to_concat.append(np.sin(2 * np.pi * dayofweek_s / 7.0).rename('dayofweek_sin'))
    new_features_to_concat.append(np.cos(2 * np.pi * dayofweek_s / 7.0).rename('dayofweek_cos'))
    new_features_to_concat.append(np.sin(2 * np.pi * dayofyear_s / 365.25).rename('dayofyear_sin'))
    new_features_to_concat.append(np.cos(2 * np.pi * dayofyear_s / 365.25).rename('dayofyear_cos'))
    
    new_features_to_concat.append(dayofweek_s.apply(lambda x: 1 if x >= 5 else 0).rename('is_weekend')) # Apply on Series
    print(f"  Created 'is_weekend' feature.")

    month_series = pd.Series(idx.month, index=idx) 
    season_cat_temp_col = 'season_cat_temp' 
    def get_season(month_val): 
        if month_val in [12, 1, 2]: return 'Winter'
        elif month_val in [3, 4, 5]: return 'Spring'
        elif month_val in [6, 7, 8]: return 'Summer'
        else: return 'Autumn' 
    
    # Add the temp categorical season to new_features_to_concat
    new_features_to_concat.append(month_series.apply(get_season).rename(season_cat_temp_col))
    # One-hot encode 'season_cat_temp' after it's part of data_feat (after first concat)
    
    # --- 5. Weather Features (Direct, Lags, Rolling) ---
    print(f"  Creating weather-based features...")
    temp_col_found = None
    possible_temp_cols = ['Temperature', 'temp', 'air_temp', 'temperature', 'Air temperature', 'T', 'tt'] 
    for col_name_df in data_feat.columns: 
        for p_name in possible_temp_cols:
            if col_name_df.strip().lower() == p_name.strip().lower():
                temp_col_found = col_name_df; break
        if temp_col_found: break
            
    weather_features_to_process = [] 
    if temp_col_found: weather_features_to_process.append(temp_col_found)
    if 'cloud_cover_percentage' in data_feat.columns: weather_features_to_process.append('cloud_cover_percentage')
    if 'visibility' in data_feat.columns: weather_features_to_process.append('visibility')
    rh_col_found = next((col for col in data_feat.columns if 'relative humidity' in col.lower()), None)
    if rh_col_found: weather_features_to_process.append(rh_col_found)

    weather_rolling_windows = [3, 6, 12, 24]; weather_lags = [1, 2, 3, 24]
    for w_feat in weather_features_to_process:
        if w_feat in data_feat.columns:
            for lag in weather_lags: 
                new_features_to_concat.append(data_feat[w_feat].shift(lag).rename(f'{w_feat}_lag_{lag}h'))
            shifted_w_feat = data_feat[w_feat].shift(1)
            for window in weather_rolling_windows:
                new_features_to_concat.append(shifted_w_feat.rolling(window=window, min_periods=1).agg("mean").rename(f'{w_feat}_roll_mean_{window}h'))
                new_features_to_concat.append(shifted_w_feat.rolling(window=window, min_periods=1).agg("std").rename(f'{w_feat}_roll_std_{window}h'))

    # --- Concatenate all collected features to data_feat ---
    if new_features_to_concat:
        data_feat = pd.concat([data_feat] + new_features_to_concat, axis=1)
    
    # --- Now that season_cat_temp is in data_feat, create dummies ---
    if season_cat_temp_col in data_feat.columns:
        season_dummies = pd.get_dummies(data_feat[season_cat_temp_col], prefix='season', dtype=int)
        data_feat = pd.concat([data_feat, season_dummies], axis=1)
        print(f"  Created one-hot encoded season features: {list(season_dummies.columns)}")
    else:
        print(f"  Warning: '{season_cat_temp_col}' not found in data_feat. Skipping season dummy creation.")


    # --- 6. Consumption per Square Meter & per Log-Square Meter Features ---
    print(f"  Creating consumption per area features...")
    consumption_per_sqm_temp_col = f'{target_building_name}_per_sqm_temp' # Temporary column name
    if building_size_col in data_feat.columns and data_feat[building_size_col].nunique() > 0 and data_feat[building_size_col].iloc[0] > 0.0001 : 
        data_feat[consumption_per_sqm_temp_col] = data_feat[target_building_name] / data_feat[building_size_col]
        data_feat[consumption_per_sqm_temp_col] = data_feat[consumption_per_sqm_temp_col].replace([np.inf, -np.inf], np.nan) # No inplace
        
        temp_consumption_features = []
        for lag in lookback_hours: 
            temp_consumption_features.append(data_feat[consumption_per_sqm_temp_col].shift(lag).rename(f'consumption_per_sqm_lag_{lag}h'))
        for window in rolling_windows:
            shifted_per_sqm = data_feat[consumption_per_sqm_temp_col].shift(1)
            for agg_func_name_str in ["mean", "std", "min", "max", "median"]:
                temp_consumption_features.append(shifted_per_sqm.rolling(window=window, min_periods=1).agg(agg_func_name_str).rename(f'consumption_per_sqm_roll_{agg_func_name_str}_{window}h'))
        if temp_consumption_features: data_feat = pd.concat([data_feat] + temp_consumption_features, axis=1)
        print(f"    Created features for consumption per raw square meter.")
    
    consumption_per_log_sqm_temp_col = f'{target_building_name}_per_log_sqm_temp' # Temporary column name
    if log_building_size_col in data_feat.columns and data_feat[log_building_size_col].nunique() > 0 and data_feat[log_building_size_col].iloc[0] > 0.0001: 
        data_feat[consumption_per_log_sqm_temp_col] = data_feat[target_building_name] / data_feat[log_building_size_col]
        data_feat[consumption_per_log_sqm_temp_col] = data_feat[consumption_per_log_sqm_temp_col].replace([np.inf, -np.inf], np.nan) # No inplace
        
        temp_log_consumption_features = []
        for lag in lookback_hours: 
            temp_log_consumption_features.append(data_feat[consumption_per_log_sqm_temp_col].shift(lag).rename(f'consumption_per_log_sqm_lag_{lag}h'))
        for window in rolling_windows:
            shifted_per_log_sqm = data_feat[consumption_per_log_sqm_temp_col].shift(1)
            for agg_func_name_str in ["mean", "std", "min", "max", "median"]:
                 temp_log_consumption_features.append(shifted_per_log_sqm.rolling(window=window, min_periods=1).agg(agg_func_name_str).rename(f'consumption_per_log_sqm_roll_{agg_func_name_str}_{window}h'))
        if temp_log_consumption_features: data_feat = pd.concat([data_feat] + temp_log_consumption_features, axis=1)
        print(f"    Created features for consumption per log-square meter.")
    
    # --- 7. Temperature Derived Features (HDD, CDD, Temp^2) ---
    base_temp_hdd_cdd = 18.0 
    temp_derived_features_to_add = {}
    if temp_col_found and temp_col_found in data_feat.columns: 
        print(f"  Creating HDD, CDD, and Temp^2 features using '{temp_col_found}'.")
        temp_series = data_feat[temp_col_found]
        temp_derived_features_to_add['hdd'] = np.maximum(0, base_temp_hdd_cdd - temp_series)
        temp_derived_features_to_add['cdd'] = np.maximum(0, temp_series - base_temp_hdd_cdd)
        temp_derived_features_to_add[f'{temp_col_found}_sq'] = temp_series ** 2
    if temp_derived_features_to_add:
        data_feat = pd.concat([data_feat] + [s.rename(name) for name, s in temp_derived_features_to_add.items()], axis=1)


    # --- 8. Interaction Features ---
    print(f"  Creating interaction features...")
    interaction_features_to_add = {} 
    if 'hour' in data_feat.columns and 'dayofweek' in data_feat.columns:
        interaction_features_to_add['hour_x_dayofweek'] = data_feat['hour'] * data_feat['dayofweek'] 
    if building_size_col in data_feat.columns:
        if 'hour_sin' in data_feat.columns: interaction_features_to_add[f'{building_size_col}_x_hour_sin'] = data_feat[building_size_col] * data_feat['hour_sin']
        if 'hour_cos' in data_feat.columns: interaction_features_to_add[f'{building_size_col}_x_hour_cos'] = data_feat[building_size_col] * data_feat['hour_cos']
        if temp_col_found and temp_col_found in data_feat.columns: interaction_features_to_add[f'{building_size_col}_x_{temp_col_found}'] = data_feat[building_size_col] * data_feat[temp_col_found]
        if 'cloud_cover_percentage' in data_feat.columns: interaction_features_to_add[f'{building_size_col}_x_cloud_cover'] = data_feat[building_size_col] * data_feat['cloud_cover_percentage']
    if log_building_size_col in data_feat.columns:
        if 'hour_sin' in data_feat.columns: interaction_features_to_add[f'{log_building_size_col}_x_hour_sin'] = data_feat[log_building_size_col] * data_feat['hour_sin']
        if 'hour_cos' in data_feat.columns: interaction_features_to_add[f'{log_building_size_col}_x_hour_cos'] = data_feat[log_building_size_col] * data_feat['hour_cos']
        if temp_col_found and temp_col_found in data_feat.columns: interaction_features_to_add[f'{log_building_size_col}_x_{temp_col_found}'] = data_feat[log_building_size_col] * data_feat[temp_col_found]
        if 'cloud_cover_percentage' in data_feat.columns: interaction_features_to_add[f'{log_building_size_col}_x_cloud_cover'] = data_feat[log_building_size_col] * data_feat['cloud_cover_percentage']
    if temp_col_found and temp_col_found in data_feat.columns:
        if 'hour' in data_feat.columns: interaction_features_to_add[f'{temp_col_found}_x_hour'] = data_feat[temp_col_found] * data_feat['hour']
        if 'is_weekend' in data_feat.columns: interaction_features_to_add[f'{temp_col_found}_x_is_weekend'] = data_feat[temp_col_found] * data_feat['is_weekend']
        for season_col_dummy in ['season_Winter', 'season_Spring', 'season_Summer', 'season_Autumn']: # Use dummy names
            if season_col_dummy in data_feat.columns:
                interaction_features_to_add[f'{temp_col_found}_x_{season_col_dummy}'] = data_feat[temp_col_found] * data_feat[season_col_dummy]
    
    if interaction_features_to_add:
        data_feat = pd.concat([data_feat] + [s.rename(name) for name, s in interaction_features_to_add.items() if name not in data_feat.columns], axis=1)
    print(f"    Created various interaction features.")
    
    # --- Finalize Feature List & Clean up ---
    cols_to_exclude_initial = [ target_building_name, 'phenomena_text', season_cat_temp_col, 
        area_bin_cat_temp_col if area_bin_cat_temp_col in data_feat.columns else None,
        consumption_per_sqm_temp_col if consumption_per_sqm_temp_col in data_feat.columns else None,
        consumption_per_log_sqm_temp_col if consumption_per_log_sqm_temp_col in data_feat.columns else None
    ]
    cols_to_exclude_initial = [col for col in cols_to_exclude_initial if col is not None and col in data_feat.columns] 
    original_cloud_col_name_check = next((c for c in data_feat.columns if 'total cloud cover' in c.lower() or c.lower() == 'c'), None)
    if original_cloud_col_name_check and original_cloud_col_name_check != 'cloud_cover_percentage': 
        cols_to_exclude_initial.append(original_cloud_col_name_check)
        
    feature_columns = [col for col in data_feat.columns if col not in cols_to_exclude_initial]
    
    print(f"  Skipping collinearity reduction step. All {len(feature_columns)} initial features will be used.")
    
    for col in feature_columns: 
        if data_feat[col].dtype == 'object':
            print(f"  Final check: Feature column '{col}' is of object type. Attempting to convert to numeric.")
            data_feat[col] = pd.to_numeric(data_feat[col], errors='coerce')

    data_feat_final = data_feat.dropna(subset=[target_building_name]).copy() 
    
    if feature_cache_dir and cache_file_df and cache_file_cols:
        try:
            df_to_save = data_feat_final.copy()
            if isinstance(df_to_save.index, pd.DatetimeIndex):
                 df_to_save = df_to_save.reset_index() 
            
            df_to_save.to_parquet(cache_file_df) 
            with open(cache_file_cols, 'w') as f:
                json.dump(feature_columns, f) 
            print(f"Saved features for {target_building_name} to cache: {cache_file_df}")
        except Exception as e:
            print(f"Warning: Could not save features for {target_building_name} to cache: {e}")

    print(f"Features created for {target_building_name}. Final shape: {data_feat_final.shape}, Number of features: {len(feature_columns)}")
    return data_feat_final, feature_columns

if __name__ == '__main__':
    print("Testing xgboost_data_preparer.py (now in models/xgboost/)...")
    current_dir_test = os.path.dirname(os.path.abspath(__file__))
    project_root_test = os.path.dirname(os.path.dirname(current_dir_test)) 
    test_cache_dir = os.path.join(project_root_test, "results", "xgboost_results", "feature_cache_test") 
    test_output_paths = {'feature_cache_dir': test_cache_dir}
    os.makedirs(test_cache_dir, exist_ok=True)

    try:
        elec_df, weather_df, areas_df = prepare_data()
        print("\nElectricity data sample:\n", elec_df.head())
        print("\nWeather data sample:\n", weather_df.head())
        
        if not elec_df.empty and not weather_df.empty:
            building_cols_test = [col for col in elec_df.columns if col not in ['time_iso', 'Datetime']]
            if building_cols_test:
                test_bldg = building_cols_test[0]
                area_map_test = areas_df.set_index('Buid_ID')['Area [m2]'].to_dict() 
                
                print("\n--- Testing feature creation (force recreate) ---")
                features_df_nocache, feature_list_nocache = create_features(
                    elec_df[[test_bldg]], weather_df.copy(), test_bldg, area_map_test, 
                    output_paths=test_output_paths, force_recreate_features=True
                )
                print(f"\nFeatures for {test_bldg} (no cache) sample (first 5 rows, 10 cols):\n", features_df_nocache.iloc[:5, :10])
                print(f"\nTotal features for {test_bldg} (no cache): {len(feature_list_nocache)}")

                print("\nChecking for new temperature-based and interaction features:")
                temp_eng_features = ['hdd', 'cdd', 'air_temperature_sq', 'air_temperature_x_is_weekend', 'air_temperature_x_season_Winter'] 
                
                actual_temp_col = None
                possible_temp_cols_test = ['Temperature', 'temp', 'air_temp', 'temperature', 'Air temperature', 'T', 'tt']
                for p_name in possible_temp_cols_test:
                    if p_name in features_df_nocache.columns: actual_temp_col = p_name; break
                    elif p_name.lower() in (c.lower() for c in features_df_nocache.columns): 
                        actual_temp_col = next(c for c in features_df_nocache.columns if c.lower() == p_name.lower())
                        break
                
                if actual_temp_col:
                    temp_eng_features = ['hdd', 'cdd', f'{actual_temp_col}_sq', f'{actual_temp_col}_x_is_weekend', f'{actual_temp_col}_x_season_Winter']

                for new_feat in temp_eng_features:
                    if new_feat in feature_list_nocache:
                        print(f"  Found new feature: {new_feat}")
                    else:
                         print(f"  MISSING expected new feature: {new_feat}")
            else:
                print("No building columns in electricity data to test feature creation.")
        else:
            print("Electricity or weather data is empty, cannot test feature creation.")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()