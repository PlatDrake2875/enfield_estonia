# models/xgboost/xgboost_pipeline.py

import os
import json 

# Import from sibling modules
from .xgboost_data_preparer import prepare_data, create_features # create_features is now imported here
from .xgboost_trainer import run_cross_validation_fold 
from .xgboost_evaluator import plot_overall_xgboost_results, save_summary_stats

# Define output directories for XGBoost results
BASE_RESULTS_DIR = "results" 
XGBOOST_RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, "xgboost_results")
XGBOOST_PLOTS_BASE_DIR = os.path.join(XGBOOST_RESULTS_DIR, "plots")
XGBOOST_DATA_DIR = os.path.join(XGBOOST_RESULTS_DIR, "data") 

# Specific subdirectories for plots
XGBOOST_FEATURE_CORRELATION_PLOTS_DIR = os.path.join(XGBOOST_PLOTS_BASE_DIR, "feature_correlations")
XGBOOST_FEATURE_TARGET_CORRELATION_PLOTS_DIR = os.path.join(XGBOOST_PLOTS_BASE_DIR, "feature_target_correlations")
XGBOOST_PREDICTION_PLOTS_DIR = os.path.join(XGBOOST_PLOTS_BASE_DIR, "prediction_plots")
XGBOOST_IMPORTANCE_PLOTS_DIR = os.path.join(XGBOOST_PLOTS_BASE_DIR, "feature_importance_plots")

# Specific subdirectories for data
XGBOOST_CORRELATION_DATA_DIR = os.path.join(XGBOOST_DATA_DIR, "correlations") 
XGBOOST_IMPORTANCE_DATA_DIR = os.path.join(XGBOOST_DATA_DIR, "feature_importance") 
XGBOOST_PREDICTIONS_DATA_DIR = os.path.join(XGBOOST_DATA_DIR, "predictions") 

XGBOOST_TENSORBOARD_LOGS_DIR = os.path.join(XGBOOST_RESULTS_DIR, "tensorboard_logs")

# --- NEW: Feature Cache Directory ---
XGBOOST_FEATURE_CACHE_DIR = os.path.join(XGBOOST_RESULTS_DIR, "feature_cache")
# --- END NEW ---


def ensure_all_output_dirs():
    """Ensures all necessary output directories for the XGBoost pipeline exist."""
    dirs_to_create = [
        XGBOOST_PLOTS_BASE_DIR, XGBOOST_FEATURE_CORRELATION_PLOTS_DIR,
        XGBOOST_FEATURE_TARGET_CORRELATION_PLOTS_DIR, XGBOOST_PREDICTION_PLOTS_DIR,
        XGBOOST_IMPORTANCE_PLOTS_DIR, XGBOOST_DATA_DIR, XGBOOST_CORRELATION_DATA_DIR,
        XGBOOST_IMPORTANCE_DATA_DIR, XGBOOST_PREDICTIONS_DATA_DIR,
        XGBOOST_TENSORBOARD_LOGS_DIR,
        XGBOOST_FEATURE_CACHE_DIR # Add Feature Cache dir
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
    print(f"All XGBoost output directories ensured under '{XGBOOST_RESULTS_DIR}'")

def main():
    """Main function to run the XGBoost Hackathon simulation pipeline."""
    print("Executing XGBoost Pipeline...") 
    ensure_all_output_dirs() 

    output_paths = {
        'plots_base_dir': XGBOOST_PLOTS_BASE_DIR,
        'feature_correlation_plots_dir': XGBOOST_FEATURE_CORRELATION_PLOTS_DIR,
        'feature_target_correlation_plots_dir': XGBOOST_FEATURE_TARGET_CORRELATION_PLOTS_DIR,
        'prediction_plots_dir': XGBOOST_PREDICTION_PLOTS_DIR,
        'importance_plots_dir': XGBOOST_IMPORTANCE_PLOTS_DIR,
        'data_dir': XGBOOST_DATA_DIR, 
        'correlation_data_dir': XGBOOST_CORRELATION_DATA_DIR,
        'importance_data_dir': XGBOOST_IMPORTANCE_DATA_DIR,
        'predictions_data_dir': XGBOOST_PREDICTIONS_DATA_DIR,
        'tensorboard_logs_dir': XGBOOST_TENSORBOARD_LOGS_DIR,
        'feature_cache_dir': XGBOOST_FEATURE_CACHE_DIR # Add to output_paths
    }

    try:
        elec_data, weather_data, areas_data = prepare_data()
    except FileNotFoundError:
        print("Halting XGBoost pipeline due to missing input files during data preparation.")
        return 
    
    if elec_data.empty:
        print("Electricity data is empty after preparation. Exiting XGBoost pipeline.")
        return

    building_names = [col for col in elec_data.columns if col not in ['time_iso', 'Datetime']]
    if not building_names:
        print("No building columns found in electricity data. Cannot proceed.")
        return
    area_map = areas_data.set_index('Buid_ID')['Area [m2]'].to_dict()
    
    cv_results_xgb = {}
    print("\nStarting XGBoost Leave-One-Out Cross-Validation within pipeline...") 
    for bldg_name in building_names:
        fold_result = run_cross_validation_fold(
            held_out_building_name=bldg_name,
            building_names=building_names,
            elec_df=elec_data,
            weather_df=weather_data,
            area_map=area_map,
            output_paths=output_paths 
        )
        if fold_result: 
            cv_results_xgb[bldg_name] = fold_result
    
    print("XGBoost LOOCV processing complete for all folds within pipeline.") 

    if cv_results_xgb: 
        plot_overall_xgboost_results(cv_results_xgb, output_paths) 
        save_summary_stats(cv_results_xgb, output_paths)
    else:
        print("No cross-validation results were generated to plot or summarize for XGBoost.")
    print("XGBoost Pipeline execution finished.") 

if __name__ == '__main__':
    print("Running XGBoost Pipeline directly (as __main__)...")
    main()
