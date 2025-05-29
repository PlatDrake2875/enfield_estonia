# xgboost_hackathon.py
# Energy Consumption Forecasting
# Section 6: Hackathon Simulation: Leave-One-Out Cross-Validation

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns 
import os 
import json 

# Define the input data directory and filenames (output from data_cleaning.py)
INPUT_DATA_DIR = 'data'
ELECTRICITY_CSV_FILENAME = 'cleaned_electricity.csv'
WEATHER_CSV_FILENAME = 'cleaned_weather.csv'
AREAS_CSV_FILENAME = 'cleaned_areas.csv'

# Define output directories for XGBoost results
BASE_RESULTS_DIR = "results" 
XGBOOST_RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, "xgboost_results")
XGBOOST_PLOTS_BASE_DIR = os.path.join(XGBOOST_RESULTS_DIR, "plots") # Base plots directory
XGBOOST_DATA_DIR = os.path.join(XGBOOST_RESULTS_DIR, "data") 

# Specific subdirectories for plots
XGBOOST_FEATURE_CORRELATION_PLOTS_DIR = os.path.join(XGBOOST_PLOTS_BASE_DIR, "feature_correlations")
XGBOOST_FEATURE_TARGET_CORRELATION_PLOTS_DIR = os.path.join(XGBOOST_PLOTS_BASE_DIR, "feature_target_correlations")
XGBOOST_PREDICTION_PLOTS_DIR = os.path.join(XGBOOST_PLOTS_BASE_DIR, "prediction_plots") # For actual vs predicted
XGBOOST_IMPORTANCE_PLOTS_DIR = os.path.join(XGBOOST_PLOTS_BASE_DIR, "feature_importance_plots") # For importance plots

# Specific subdirectories for data
XGBOOST_CORRELATION_DATA_DIR = os.path.join(XGBOOST_DATA_DIR, "correlations") 
XGBOOST_IMPORTANCE_DATA_DIR = os.path.join(XGBOOST_DATA_DIR, "feature_importance") 
XGBOOST_PREDICTIONS_DATA_DIR = os.path.join(XGBOOST_DATA_DIR, "predictions") # For prediction CSVs

FIGURE_DPI = 300

def ensure_xgboost_output_dirs():
    """Creates output directories for XGBoost results if they don't exist."""
    os.makedirs(XGBOOST_PLOTS_BASE_DIR, exist_ok=True)
    os.makedirs(XGBOOST_FEATURE_CORRELATION_PLOTS_DIR, exist_ok=True)
    os.makedirs(XGBOOST_FEATURE_TARGET_CORRELATION_PLOTS_DIR, exist_ok=True)
    os.makedirs(XGBOOST_PREDICTION_PLOTS_DIR, exist_ok=True)
    os.makedirs(XGBOOST_IMPORTANCE_PLOTS_DIR, exist_ok=True)
    
    os.makedirs(XGBOOST_DATA_DIR, exist_ok=True)
    os.makedirs(XGBOOST_CORRELATION_DATA_DIR, exist_ok=True) 
    os.makedirs(XGBOOST_IMPORTANCE_DATA_DIR, exist_ok=True) 
    os.makedirs(XGBOOST_PREDICTIONS_DATA_DIR, exist_ok=True)
    
    print(f"XGBoost output directories ensured under '{XGBOOST_RESULTS_DIR}'")


def prepare_data():
    """Load and prepare data from cleaned CSV files."""
    print("Preparing data for XGBoost Hackathon...")
    
    electricity_path = os.path.join(INPUT_DATA_DIR, ELECTRICITY_CSV_FILENAME)
    weather_path = os.path.join(INPUT_DATA_DIR, WEATHER_CSV_FILENAME)
    areas_path = os.path.join(INPUT_DATA_DIR, AREAS_CSV_FILENAME)

    try:
        elec = pd.read_csv(electricity_path)
        weather = pd.read_csv(weather_path)
        areas = pd.read_csv(areas_path)
    except FileNotFoundError as e:
        print(f"Error: One or more cleaned CSV files not found in '{INPUT_DATA_DIR}'. Please run data_cleaning.py first.")
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


def hackathon_split(data_with_features, target_building_col, train_months=2):
    """Splits data for the hackathon scenario (2 months train, rest test for the held-out building)."""
    print(f"Splitting data for {target_building_col} (hackathon setup)...")
    data_with_features = data_with_features.sort_index()
    
    if data_with_features.empty:
        print(f"  No data to split for {target_building_col}.")
        return pd.DataFrame(), pd.DataFrame()

    start_date = data_with_features.index.min()
    split_date = start_date + pd.DateOffset(months=train_months)
    
    train_df = data_with_features[data_with_features.index < split_date]
    test_df = data_with_features[data_with_features.index >= split_date]
    
    if train_df.empty or test_df.empty:
        print(f"  Warning: Not enough data to create a valid train/test split for {target_building_col} after {train_months} months.")
        return pd.DataFrame(), pd.DataFrame()
        
    print(f"  Split for {target_building_col}: Train shape {train_df.shape}, Test shape {test_df.shape}")
    return train_df, test_df


def cross_validate_buildings_xgboost(elec_df, weather_df, areas_df):
    """Performs leave-one-out cross-validation using XGBoost."""
    print("\nStarting XGBoost Leave-One-Out Cross-Validation...")
    ensure_xgboost_output_dirs()

    building_names = [col for col in elec_df.columns if col not in ['time_iso', 'Datetime']] 
    if not building_names:
        print("No building columns found in electricity data. Cannot proceed with XGBoost CV.")
        return {}
        
    area_map = areas_df.set_index('Buid_ID')['Area [m2]'].to_dict()
    results_cv = {}

    for held_out_building_name in building_names:
        print(f"\n--- Processing with {held_out_building_name} as the held-out (new) building ---")
        
        train_building_names = [b for b in building_names if b != held_out_building_name]
        X_train_from_others_list, y_train_from_others_list = [], []
        common_feature_set = None 

        for train_bldg_name in train_building_names:
            data_train_bldg_feat, features_train_bldg = create_features(
                elec_df[[train_bldg_name]], weather_df.copy(), train_bldg_name, area_map
            )
            if not data_train_bldg_feat.empty and train_bldg_name in data_train_bldg_feat.columns:
                X_train_from_others_list.append(data_train_bldg_feat[features_train_bldg])
                y_train_from_others_list.append(data_train_bldg_feat[train_bldg_name])
                
                if common_feature_set is None:
                    common_feature_set = set(features_train_bldg)
                else:
                    common_feature_set = common_feature_set.intersection(set(features_train_bldg))

        if not X_train_from_others_list or common_feature_set is None:
            print(f"  No training data or common features from other buildings for held-out {held_out_building_name}. Skipping.")
            continue
        
        common_feature_list = sorted(list(common_feature_set))
        if not common_feature_list:
            print(f"  No common features list found across training buildings for held-out {held_out_building_name}. Skipping.")
            continue

        X_train_all_others = pd.concat([df[common_feature_list] for df in X_train_from_others_list if not df[common_feature_list].empty])
        y_train_all_others = pd.concat(y_train_from_others_list)
        if X_train_all_others.empty:
            print(f"  Concatenated training data from other buildings is empty for {held_out_building_name}. Skipping.")
            continue

        data_held_out_feat, features_held_out = create_features(
            elec_df[[held_out_building_name]], weather_df.copy(), held_out_building_name, area_map
        )
        
        if data_held_out_feat.empty or held_out_building_name not in data_held_out_feat.columns:
            print(f"  No data or target column after feature creation for held-out building {held_out_building_name}. Skipping.")
            continue
        
        train_context_held_out, test_target_held_out = hackathon_split(data_held_out_feat, held_out_building_name)

        if train_context_held_out.empty or test_target_held_out.empty:
            print(f"  Not enough data to split for held-out building {held_out_building_name}. Skipping.")
            continue
            
        final_common_features = sorted(list(set(common_feature_list) & set(features_held_out)))
        if not final_common_features:
            print(f"  No common features between 'other buildings' and 'held-out context' for {held_out_building_name}. Skipping.")
            continue

        X_train_final = pd.concat([
            X_train_all_others[final_common_features], 
            train_context_held_out[final_common_features]
        ])
        y_train_final = pd.concat([
            y_train_all_others, 
            train_context_held_out[held_out_building_name]
        ])
        
        X_test_final = test_target_held_out[final_common_features]
        y_test_final = test_target_held_out[held_out_building_name]

        if X_train_final.empty or X_test_final.empty:
            print(f"  Training or Test data is empty for {held_out_building_name} before XGBoost. Skipping.")
            continue

        # --- Correlation Analysis for this fold ---
        print(f"  Performing correlation analysis for fold: {held_out_building_name}")
        if not X_train_final.empty:
            # Feature-Feature Correlation
            corr_matrix = X_train_final.corr()
            corr_matrix_filename = os.path.join(XGBOOST_CORRELATION_DATA_DIR, f"feature_correlation_matrix_{held_out_building_name}.csv") # Save CSV to data subdir
            corr_matrix.to_csv(corr_matrix_filename)
            print(f"    Feature correlation matrix saved to {corr_matrix_filename}")

            fig_corr, ax_corr = plt.subplots(figsize=(max(12, len(final_common_features)*0.6), max(10, len(final_common_features)*0.5)), dpi=FIGURE_DPI)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr, 
                        annot_kws={"size": 6 if len(final_common_features) < 25 else 4}) 
            ax_corr.set_title(f"Feature Correlation Matrix (Train data for {held_out_building_name} fold)", fontsize=16)
            plt.tight_layout(pad=1.5)
            heatmap_filename = os.path.join(XGBOOST_FEATURE_CORRELATION_PLOTS_DIR, f"feature_correlation_heatmap_{held_out_building_name}.png") # Save plot to specific plot subdir
            plt.savefig(heatmap_filename)
            print(f"    Feature correlation heatmap saved to {heatmap_filename}")
            plt.close(fig_corr)

            # Feature-Target Correlation
            combined_train_df_for_corr = X_train_final.copy()
            combined_train_df_for_corr['TARGET'] = y_train_final.values 
            
            if 'TARGET' in combined_train_df_for_corr.columns:
                target_corr = combined_train_df_for_corr.corr()['TARGET'].drop('TARGET').sort_values(ascending=False)
                target_corr_filename = os.path.join(XGBOOST_CORRELATION_DATA_DIR, f"feature_target_correlation_{held_out_building_name}.csv") # Save CSV to data subdir
                target_corr.to_csv(target_corr_filename)
                print(f"    Feature-target correlation saved to {target_corr_filename}")

                fig_target_corr, ax_target_corr = plt.subplots(figsize=(12, max(8, len(target_corr)*0.35)), dpi=FIGURE_DPI)
                target_corr.plot(kind='barh', ax=ax_target_corr, color=sns.color_palette("coolwarm_r", len(target_corr)))
                ax_target_corr.set_title(f"Feature Correlation with Target (Train data for {held_out_building_name} fold)", fontsize=16)
                ax_target_corr.set_xlabel("Pearson Correlation", fontsize=12)
                plt.tight_layout(pad=1.5)
                target_corr_plot_filename = os.path.join(XGBOOST_FEATURE_TARGET_CORRELATION_PLOTS_DIR, f"feature_target_correlation_barchart_{held_out_building_name}.png") # Save plot to specific plot subdir
                plt.savefig(target_corr_plot_filename)
                print(f"    Feature-target correlation plot saved to {target_corr_plot_filename}")
                plt.close(fig_target_corr)
            else:
                print("    Could not compute feature-target correlation: 'TARGET' column missing after merge.")
        # --- End Correlation Analysis ---


        X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
            X_train_final, y_train_final, test_size=0.2, random_state=42, shuffle=False 
        )

        print(f"  Final Train shapes for XGB: X={X_train_xgb.shape}, y={y_train_xgb.shape}")
        print(f"  Final Val shapes for XGB: X={X_val_xgb.shape}, y={y_val_xgb.shape}")
        print(f"  Final Test shapes for XGB: X={X_test_final.shape}, y={y_test_final.shape}")

        dtrain_xgb = xgb.DMatrix(X_train_xgb, label=y_train_xgb, feature_names=final_common_features)
        dval_xgb = xgb.DMatrix(X_val_xgb, label=y_val_xgb, feature_names=final_common_features)
        dtest_xgb = xgb.DMatrix(X_test_final, label=y_test_final, feature_names=final_common_features)
        
        params_xgb = {
            'objective': 'reg:squarederror', 
            'eval_metric': 'rmse', 
            'eta': 0.03, 
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
        }

        print(f"  Training XGBoost for held-out building: {held_out_building_name}...")
        evals_result_xgb = {} 
        model_xgb = xgb.train(
            params_xgb,
            dtrain_xgb,
            num_boost_round=1000, 
            evals=[(dtrain_xgb, 'train'), (dval_xgb, 'eval')],
            evals_result=evals_result_xgb, 
            early_stopping_rounds=100, 
            verbose_eval=100, 
        )
        
        predictions_xgb = model_xgb.predict(dtest_xgb)
        
        y_test_final_mape = y_test_final.copy()
        y_test_final_mape[y_test_final_mape == 0] = 0.001 

        mape_xgb = mean_absolute_percentage_error(y_test_final_mape, predictions_xgb) * 100
        
        fold_importance_scores = {}
        for imp_type in ['weight', 'gain', 'cover']:
            try:
                scores = model_xgb.get_score(importance_type=imp_type)
                fold_importance_scores[imp_type] = scores
            except Exception as e_imp_score:
                print(f"    Could not get '{imp_type}' importance for {held_out_building_name}: {e_imp_score}")
                fold_importance_scores[imp_type] = {}
        
        importance_df = pd.DataFrame.from_dict({(imp_type, feat): score 
                                                for imp_type, scores_dict in fold_importance_scores.items() 
                                                for feat, score in scores_dict.items()},
                                               orient='index', columns=['score'])
        importance_df.index = pd.MultiIndex.from_tuples(importance_df.index, names=['importance_type', 'feature'])
        importance_df = importance_df.unstack(level='importance_type').fillna(0)
        if not importance_df.empty: 
             if 'score' in importance_df.columns: 
                importance_df.columns = importance_df.columns.droplevel(0) 
        
        importance_filename = os.path.join(XGBOOST_IMPORTANCE_DATA_DIR, f"feature_importance_scores_{held_out_building_name}.csv") # Save CSV to data subdir
        importance_df.to_csv(importance_filename)
        print(f"    Feature importance scores for fold saved to {importance_filename}")

        results_cv[held_out_building_name] = {
            'mape': mape_xgb,
            'true': y_test_final.values,
            'pred': predictions_xgb,
            'dates': y_test_final.index,
            'model': model_xgb, 
            'evals_result': evals_result_xgb, 
            'feature_names': final_common_features,
            'fold_importance_scores': fold_importance_scores 
        }
        print(f"  MAPE for {held_out_building_name}: {mape_xgb:.2f}%")
        
        pred_df = pd.DataFrame({'dates': y_test_final.index, 'true': y_test_final.values, 'predicted': predictions_xgb})
        pred_filename = os.path.join(XGBOOST_PREDICTIONS_DATA_DIR, f"predictions_{held_out_building_name}.csv") # Save CSV to data subdir
        pred_df.to_csv(pred_filename, index=False)
        print(f"  Predictions saved to {pred_filename}")

    print("XGBoost LOOCV complete.")
    return results_cv


def plot_xgboost_results(cv_results):
    """Plots MAPE comparison, actual vs. predicted for best, and feature importances for best. Saves plots."""
    if not cv_results:
        print("No XGBoost CV results to plot.")
        return

    print("\nPlotting XGBoost LOOCV Results...")
    
    # Plot 1: MAPE comparison
    fig_mape, ax_mape = plt.subplots(figsize=(16, 8), dpi=FIGURE_DPI) 
    mapes_cv = {k: v['mape'] for k, v in cv_results.items() if 'mape' in v and pd.notna(v['mape'])}
    
    if not mapes_cv:
        print("No valid MAPE results to plot.")
        plt.close(fig_mape) 
        return

    sorted_mapes = sorted(mapes_cv.items(), key=lambda item: item[1])
    buildings_sorted = [item[0] for item in sorted_mapes]
    mape_values_sorted = [item[1] for item in sorted_mapes]

    bars = ax_mape.bar(buildings_sorted, mape_values_sorted, color=sns.color_palette("viridis", len(buildings_sorted)))
    ax_mape.tick_params(axis='x', labelsize=10) 
    ax_mape.set_xticks(ax_mape.get_xticks()) 
    ax_mape.set_xticklabels(ax_mape.get_xticklabels(), rotation=45, ha="right")

    ax_mape.tick_params(axis='y', labelsize=10)
    ax_mape.set_title('XGBoost LOOCV: MAPE by Held-Out Building', fontsize=18) 
    ax_mape.set_ylabel('MAPE (%)', fontsize=14) 
    for bar in bars:
        height = bar.get_height()
        ax_mape.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout(pad=1.5) 
    plot_filename_mape = os.path.join(XGBOOST_PLOTS_BASE_DIR, "xgboost_loocv_mape_comparison.png") # Save to base plots dir
    plt.savefig(plot_filename_mape)
    print(f"MAPE comparison plot saved to {plot_filename_mape}")
    plt.close(fig_mape) 
    
    if sorted_mapes:
        best_bldg_cv = sorted_mapes[0][0]
        result_best = cv_results[best_bldg_cv]
        
        fig_pred, ax_pred = plt.subplots(figsize=(20, 8), dpi=FIGURE_DPI) 
        ax_pred.plot(result_best['dates'], result_best['true'], label='Actual', alpha=0.8, color='blue', linewidth=1.5)
        ax_pred.plot(result_best['dates'], result_best['pred'], label='Predicted (XGBoost)', alpha=0.8, linestyle='--', color='red', linewidth=1.5)
        ax_pred.set_title(f'XGBoost Predictions for {best_bldg_cv} (Best MAPE: {result_best["mape"]:.2f}%)', fontsize=18)
        ax_pred.set_xlabel("Date", fontsize=14)
        ax_pred.set_ylabel("kWh", fontsize=14)
        ax_pred.legend(fontsize='large') 
        ax_pred.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout(pad=1.5)
        plot_filename_pred = os.path.join(XGBOOST_PREDICTION_PLOTS_DIR, f"xgboost_predictions_{best_bldg_cv}_best.png") # Save to specific plot subdir
        plt.savefig(plot_filename_pred)
        print(f"Best building prediction plot saved to {plot_filename_pred}")
        plt.close(fig_pred)

        if 'model' in result_best and hasattr(result_best['model'], 'get_score') and result_best['feature_names']:
            if not result_best['feature_names']:
                print(f"No feature names available for importance plot of {best_bldg_cv}.")
            else:
                importance_types = ['weight', 'gain', 'cover']
                for imp_type in importance_types:
                    num_features_to_plot = min(30, len(result_best['feature_names']))
                    fig_height_importance = max(8, num_features_to_plot * 0.4) 
                    
                    fig_importance, ax_importance = plt.subplots(figsize=(14, fig_height_importance), dpi=FIGURE_DPI) 
                    try:
                        xgb.plot_importance(result_best['model'], ax=ax_importance, 
                                            importance_type=imp_type, 
                                            max_num_features=num_features_to_plot, height=0.8,
                                            title=f'{imp_type.capitalize()} Importance - {best_bldg_cv}') 
                        ax_importance.tick_params(labelsize=10)
                        plt.tight_layout(pad=1.5) 
                        plot_filename_importance = os.path.join(XGBOOST_IMPORTANCE_PLOTS_DIR, f"xgboost_feature_importance_{imp_type}_{best_bldg_cv}_best.png") # Save to specific plot subdir
                        plt.savefig(plot_filename_importance)
                        print(f"Feature importance plot ({imp_type}) saved to {plot_filename_importance}")
                    except Exception as e_imp:
                        print(f"Could not plot feature importance ({imp_type}) for {best_bldg_cv}: {e_imp}")
                    finally:
                        plt.close(fig_importance) 
    else:
        print("No best building to plot predictions for.")


def main():
    """Main function to run XGBoost Hackathon simulation."""
    try:
        elec_data, weather_data, areas_data = prepare_data()
    except FileNotFoundError:
        print("Halting XGBoost simulation due to missing input files.")
        return 
    
    if elec_data.empty:
        print("Electricity data is empty after preparation. Exiting XGBoost simulation.")
        return

    cv_results_xgb = cross_validate_buildings_xgboost(elec_data, weather_data, areas_data)
    
    if cv_results_xgb: 
        plot_xgboost_results(cv_results_xgb) 
        all_mapes = [res['mape'] for res in cv_results_xgb.values() if 'mape' in res and pd.notna(res['mape'])]
        if all_mapes:
            avg_mape = np.mean(all_mapes)
            print(f"\nOverall Average MAPE from XGBoost LOOCV: {avg_mape:.2f}%")
            summary_stats = {"average_mape": avg_mape, "individual_mapes": {k:v['mape'] for k,v in cv_results_xgb.items() if 'mape' in v and pd.notna(v['mape'])}}
            summary_filename = os.path.join(XGBOOST_DATA_DIR, "xgboost_loocv_summary.json") # Summary JSON in base data dir
            try:
                with open(summary_filename, 'w') as f:
                    json.dump(summary_stats, f, indent=4)
                print(f"Overall summary saved to {summary_filename}")
            except Exception as e:
                print(f"Error saving summary JSON: {e}")
        else:
            print("No valid MAPE scores to average.")
    else:
        print("No cross-validation results to plot or average for XGBoost.")

if __name__ == '__main__':
    main()
