# models/xgboost/predict_new_building.py

import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


# Assuming these sibling modules are in the same 'models/xgboost/' directory
from .xgboost_data_preparer import prepare_data as load_cleaned_weather_and_areas, create_features
# Note: prepare_data from xgboost_data_preparer loads weather and areas, not electricity for new building

# --- Configuration ---
NEW_BUILDING_ENERGY_FILE = 'new_building_energy.csv' # Expected in project root or specify full path
NEW_BUILDING_NAME = 'NEW_BLDG_001' # Example name for the new building
NEW_BUILDING_AREA_SQM = 1000.0 # Example area, replace with actual

# Define start and end for the known 2-month period for the new building
KNOWN_DATA_START_DATE = '2023-01-01'
KNOWN_DATA_END_DATE = '2023-02-28' # End of February for 2 full months

# Output directories (relative to where this script might be called from, or adjust as needed)
BASE_RESULTS_DIR_PREDICT = "results"
NEW_BUILDING_RESULTS_DIR = os.path.join(BASE_RESULTS_DIR_PREDICT, "new_building_predictions", NEW_BUILDING_NAME)
NEW_BUILDING_DATA_DIR = os.path.join(NEW_BUILDING_RESULTS_DIR, "data")
NEW_BUILDING_MODELS_DIR = os.path.join(NEW_BUILDING_RESULTS_DIR, "finetuned_models") # To save finetuned models

# Paths to existing LOOCV model artifacts (assuming structure from previous scripts)
XGBOOST_RESULTS_DIR_ORIGINAL = os.path.join(BASE_RESULTS_DIR, "xgboost_results")
ORIGINAL_TRAINED_MODELS_DIR = os.path.join(XGBOOST_RESULTS_DIR_ORIGINAL, "data", "trained_models")
ORIGINAL_FEATURE_CACHE_DIR = os.path.join(XGBOOST_RESULTS_DIR_ORIGINAL, "feature_cache") # For create_features

FIGURE_DPI = 150 # For any plots, though not primary focus here

def ensure_output_dirs_for_new_building():
    """Creates output directories for the new building's results."""
    os.makedirs(NEW_BUILDING_DATA_DIR, exist_ok=True)
    os.makedirs(NEW_BUILDING_MODELS_DIR, exist_ok=True)
    print(f"Output directories for new building '{NEW_BUILDING_NAME}' ensured under '{NEW_BUILDING_RESULTS_DIR}'")

def load_new_building_energy_data(file_path, start_date_str, end_date_str):
    """Loads and prepares the 2-month energy data for the new building."""
    print(f"Loading energy data for new building from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        if 'time_iso' not in df.columns or len(df.columns) < 2: # Expect time_iso and one energy column
            raise ValueError("New building energy CSV must contain 'time_iso' and an energy column.")
        
        energy_col = df.columns[1] # Assume second column is energy
        df.rename(columns={energy_col: NEW_BUILDING_NAME}, inplace=True)
        
        df['Datetime'] = pd.to_datetime(df['time_iso'])
        df = df.set_index('Datetime')
        df = df[[NEW_BUILDING_NAME]] # Keep only the target column
        
        # Filter for the specified 2-month period
        start_dt = pd.to_datetime(start_date_str)
        end_dt = pd.to_datetime(end_date_str) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) # Inclusive end
        
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        # Resample to hourly if not already, and fill missing values
        df = df.resample('h').mean() # Assuming mean aggregation is okay
        df.interpolate(method='time', inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        if df.empty:
            raise ValueError(f"No data found for the new building in the period {start_date_str} to {end_date_str}.")
            
        print(f"  Loaded {len(df)} hourly records for new building '{NEW_BUILDING_NAME}'.")
        return df
    except FileNotFoundError:
        print(f"Error: New building energy file not found at '{file_path}'.")
        raise
    except Exception as e:
        print(f"Error loading new building energy data: {e}")
        raise

def finetune_and_predict():
    """
    Loads LOOCV models, finetunes them with new building data, and predicts future consumption.
    """
    ensure_output_dirs_for_new_building()

    # 1. Load Weather and All Areas Data (for feature engineering context if needed)
    try:
        # prepare_data loads weather for the whole period defined in its source CSV
        # and areas for all original buildings.
        _, weather_df_full, areas_df_all = load_cleaned_weather_and_areas()
        if weather_df_full is None or weather_df_full.empty:
            print("Critical error: Full weather data could not be loaded. Cannot proceed.")
            return
    except Exception as e:
        print(f"Error loading common weather/areas data: {e}")
        return

    # 2. Load New Building's Initial 2-Month Energy Data
    # Construct path assuming NEW_BUILDING_ENERGY_FILE is in project root
    current_script_dir = os.path.dirname(os.path.abspath(__file__)) # models/xgboost
    models_dir = os.path.dirname(current_script_dir) # models
    project_root = os.path.dirname(models_dir) # Enfield_29_05
    new_building_energy_filepath = os.path.join(project_root, NEW_BUILDING_ENERGY_FILE)

    try:
        new_bldg_energy_2months = load_new_building_energy_data(new_building_energy_filepath, KNOWN_DATA_START_DATE, KNOWN_DATA_END_DATE)
    except Exception as e:
        print(f"Could not load new building energy data: {e}. Halting.")
        return

    # 3. Define time ranges
    finetune_start_dt = pd.to_datetime(KNOWN_DATA_START_DATE)
    finetune_end_dt = pd.to_datetime(KNOWN_DATA_END_DATE) 
    
    predict_start_dt = finetune_end_dt + pd.Timedelta(hours=1) # Start predicting from the hour after known data ends
    predict_end_dt = finetune_end_dt + pd.DateOffset(months=10) # Predict for 10 months
    
    print(f"Finetuning period: {finetune_start_dt} to {finetune_end_dt}")
    print(f"Prediction period: {predict_start_dt} to {predict_end_dt}")

    # 4. Prepare features for the new building for the entire period (2 months + 10 months)
    # This ensures all lags and rolling windows can be calculated correctly.
    # We'll slice this later for finetuning and prediction.
    
    # Create a dummy area_map for the new building for create_features
    new_bldg_area_map = {NEW_BUILDING_NAME: NEW_BUILDING_AREA_SQM}
    
    # We need a DataFrame for the new building that spans the full 12 months for feature creation
    # The energy data for the prediction period will be NaN, which is fine as it's the target.
    full_period_index = pd.date_range(start=finetune_start_dt, end=predict_end_dt, freq='h')
    new_bldg_df_full_period = pd.DataFrame(index=full_period_index)
    new_bldg_df_full_period[NEW_BUILDING_NAME] = np.nan
    new_bldg_df_full_period.loc[new_bldg_energy_2months.index, NEW_BUILDING_NAME] = new_bldg_energy_2months[NEW_BUILDING_NAME]

    # Output paths for create_features caching (can reuse original cache dir or a new one)
    # For simplicity, let's assume create_features can handle its own caching if output_paths is passed
    # We need to provide the path where create_features expects to find its cache
    feature_creation_output_paths = {'feature_cache_dir': ORIGINAL_FEATURE_CACHE_DIR}


    print(f"Creating features for new building '{NEW_BUILDING_NAME}' for the full period...")
    all_features_new_bldg_df, all_feature_names_new_bldg = create_features(
        df_elec_single_building=new_bldg_df_full_period[[NEW_BUILDING_NAME]], # Pass the target column
        df_weather_full=weather_df_full.copy(), # Pass full weather data
        target_building_name=NEW_BUILDING_NAME,
        area_map_dict=new_bldg_area_map,
        output_paths=feature_creation_output_paths, # For caching
        force_recreate_features=False # Use cache if available for new building's features
    )
    
    # Slice for finetuning and prediction feature sets
    X_finetune_new_bldg_all_feats = all_features_new_bldg_df.loc[finetune_start_dt:finetune_end_dt, all_feature_names_new_bldg]
    y_finetune_new_bldg = new_bldg_energy_2months.loc[X_finetune_new_bldg_all_feats.index, NEW_BUILDING_NAME] # Align y with X's index

    X_predict_new_bldg_all_feats = all_features_new_bldg_df.loc[predict_start_dt:predict_end_dt, all_feature_names_new_bldg]

    all_predictions = {}

    # 5. Iterate through saved LOOCV models
    if not os.path.exists(ORIGINAL_TRAINED_MODELS_DIR):
        print(f"Error: Original trained models directory not found at {ORIGINAL_TRAINED_MODELS_DIR}")
        return

    for model_file in os.listdir(ORIGINAL_TRAINED_MODELS_DIR):
        if model_file.endswith(".json") and model_file.startswith("xgb_model_"):
            original_model_path = os.path.join(ORIGINAL_TRAINED_MODELS_DIR, model_file)
            print(f"\nProcessing original model: {model_file}")

            try:
                # Load the original model
                original_model = xgb.Booster()
                original_model.load_model(original_model_path)
                
                # Get feature names the original model was trained on
                model_specific_features = original_model.feature_names
                if not model_specific_features:
                    print(f"  Warning: Could not retrieve feature names from model {model_file}. Skipping this model.")
                    continue
                
                print(f"  Original model trained on {len(model_specific_features)} features.")

                # Prepare new building's data with this specific feature set
                X_finetune_current_model = X_finetune_new_bldg_all_feats[model_specific_features]
                # y_finetune_new_bldg is already prepared and aligned
                
                dtrain_finetune = xgb.DMatrix(X_finetune_current_model, label=y_finetune_new_bldg, feature_names=model_specific_features)

                # Finetune (continue training)
                print(f"  Finetuning model {model_file} with 2 months of data from '{NEW_BUILDING_NAME}'...")
                params = original_model.save_config() # Get original params
                params = json.loads(params)
                # Update relevant params for finetuning if needed, e.g., learning rate could be smaller
                # For now, use original params + xgb_model for continuation
                
                finetuned_model = xgb.train(
                    params=params.get('learner', {}).get('learner_train_param', {}), # Extract learner params
                    dtrain=dtrain_finetune,
                    num_boost_round=100,  # Number of additional boosting rounds for finetuning
                    xgb_model=original_model, # Start training from the loaded model
                    verbose_eval=50
                )
                
                # Save the finetuned model
                finetuned_model_filename = os.path.join(NEW_BUILDING_MODELS_DIR, f"finetuned_{model_file}")
                finetuned_model.save_model(finetuned_model_filename)
                print(f"  Finetuned model saved to: {finetuned_model_filename}")

                # Prepare prediction data with model-specific features
                X_predict_current_model = X_predict_new_bldg_all_feats[model_specific_features]
                dpredict_new_bldg = xgb.DMatrix(X_predict_current_model, feature_names=model_specific_features)
                
                # Predict for the next 10 months
                predictions = finetuned_model.predict(dpredict_new_bldg)
                
                fold_name = model_file.replace("xgb_model_", "").replace("_fold.json", "")
                all_predictions[f"predictions_from_finetuned_{fold_name}"] = predictions
                print(f"  Generated 10-month predictions for '{NEW_BUILDING_NAME}' using finetuned model from {fold_name} fold.")

            except Exception as e_fold:
                print(f"  Error processing model {model_file}: {e_fold}")
                import traceback
                traceback.print_exc()


    # 6. Aggregate and Save Predictions
    if not all_predictions:
        print("No predictions were generated.")
        return

    predictions_df = pd.DataFrame(all_predictions, index=X_predict_new_bldg_all_feats.index)
    
    # Calculate ensemble mean prediction
    predictions_df['ensemble_mean_prediction'] = predictions_df.mean(axis=1)
    
    # Add actual known values for the first 2 months for context if needed (optional)
    # For now, just saving the 10-month forecast
    
    output_predictions_csv = os.path.join(NEW_BUILDING_DATA_DIR, f"predictions_10months_{NEW_BUILDING_NAME}.csv")
    predictions_df.to_csv(output_predictions_csv)
    print(f"\nAll 10-month predictions for '{NEW_BUILDING_NAME}' saved to: {output_predictions_csv}")

    print(f"Predictions DataFrame head:\n{predictions_df.head()}")


if __name__ == '__main__':
    # This script is intended to be run after the main LOOCV pipeline has
    # generated cleaned data and trained models.
    # You would need to:
    # 1. Place `new_building_energy.csv` in your project root.
    # 2. Update NEW_BUILDING_NAME and NEW_BUILDING_AREA_SQM at the top of this script.
    print(f"Running prediction pipeline for new building: {NEW_BUILDING_NAME}")
    
    # Create a dummy new_building_energy.csv for testing if it doesn't exist
    dummy_file_path = NEW_BUILDING_ENERGY_FILE
    if not os.path.exists(dummy_file_path):
        print(f"Creating dummy '{dummy_file_path}' for testing purposes...")
        dummy_start = pd.to_datetime(KNOWN_DATA_START_DATE)
        dummy_end = pd.to_datetime(KNOWN_DATA_END_DATE)
        dummy_index = pd.date_range(start=dummy_start, end=dummy_end, freq='h')
        dummy_data = {
            'time_iso': [dt.strftime('%Y-%m-%dT%H:%M:%S') for dt in dummy_index],
            'Energy_kWh': np.random.rand(len(dummy_index)) * 50 + 10 # Random data
        }
        pd.DataFrame(dummy_data).to_csv(dummy_file_path, index=False)
        print(f"Dummy file '{dummy_file_path}' created.")

    finetune_and_predict()
