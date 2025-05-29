# models/xgboost/xgboost_trainer.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import math
import seaborn as sns
import os

# Import from sibling module
from .xgboost_data_preparer import create_features 

# --- NEW IMPORT for TensorBoard ---
try:
    from xgboost.callback import TensorBoard
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: xgboost.callback.TensorBoard could not be imported. TensorBoard logging will be disabled. Ensure TensorFlow is installed if you want TensorBoard logging.")
    TENSORBOARD_AVAILABLE = False
# --- END NEW IMPORT ---


FIGURE_DPI = 300 

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

def run_cross_validation_fold(
    held_out_building_name, 
    building_names, 
    elec_df, 
    weather_df, 
    area_map,
    output_paths # Dictionary containing paths for saving outputs
    ):
    """Runs a single fold of the leave-one-out cross-validation."""
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
        print(f"  No training data or common features from other buildings for held-out {held_out_building_name}. Skipping fold.")
        return None
    
    common_feature_list = sorted(list(common_feature_set))
    if not common_feature_list:
        print(f"  No common features list found across training buildings for held-out {held_out_building_name}. Skipping fold.")
        return None

    X_train_all_others = pd.concat([df[common_feature_list] for df in X_train_from_others_list if not df[common_feature_list].empty])
    y_train_all_others = pd.concat(y_train_from_others_list)
    if X_train_all_others.empty:
        print(f"  Concatenated training data from other buildings is empty for {held_out_building_name}. Skipping fold.")
        return None

    data_held_out_feat, features_held_out = create_features(
        elec_df[[held_out_building_name]], weather_df.copy(), held_out_building_name, area_map
    )
    
    if data_held_out_feat.empty or held_out_building_name not in data_held_out_feat.columns:
        print(f"  No data or target column after feature creation for held-out building {held_out_building_name}. Skipping fold.")
        return None
    
    train_context_held_out, test_target_held_out = hackathon_split(data_held_out_feat, held_out_building_name)

    if train_context_held_out.empty or test_target_held_out.empty:
        print(f"  Not enough data to split for held-out building {held_out_building_name}. Skipping fold.")
        return None
        
    final_common_features = sorted(list(set(common_feature_list) & set(features_held_out)))
    if not final_common_features:
        print(f"  No common features between 'other buildings' and 'held-out context' for {held_out_building_name}. Skipping fold.")
        return None

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
        print(f"  Training or Test data is empty for {held_out_building_name} before XGBoost. Skipping fold.")
        return None

    # Correlation Analysis (remains the same)
    print(f"  Performing correlation analysis for fold: {held_out_building_name}")
    if not X_train_final.empty:
        corr_matrix = X_train_final.corr()
        corr_matrix_filename = os.path.join(output_paths['correlation_data_dir'], f"feature_correlation_matrix_{held_out_building_name}.csv")
        corr_matrix.to_csv(corr_matrix_filename)
        print(f"    Feature correlation matrix saved to {corr_matrix_filename}")

        fig_corr, ax_corr = plt.subplots(figsize=(max(12, len(final_common_features)*0.6), max(10, len(final_common_features)*0.5)), dpi=FIGURE_DPI)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr, 
                    annot_kws={"size": 6 if len(final_common_features) < 25 else 4}) 
        ax_corr.set_title(f"Feature Correlation Matrix (Train data for {held_out_building_name} fold)", fontsize=16)
        plt.tight_layout(pad=1.5)
        heatmap_filename = os.path.join(output_paths['feature_correlation_plots_dir'], f"feature_correlation_heatmap_{held_out_building_name}.png")
        plt.savefig(heatmap_filename)
        print(f"    Feature correlation heatmap saved to {heatmap_filename}")
        plt.close(fig_corr)

        combined_train_df_for_corr = X_train_final.copy()
        combined_train_df_for_corr['TARGET'] = y_train_final.values 
        
        if 'TARGET' in combined_train_df_for_corr.columns:
            target_corr = combined_train_df_for_corr.corr()['TARGET'].drop('TARGET').sort_values(ascending=False)
            target_corr_filename = os.path.join(output_paths['correlation_data_dir'], f"feature_target_correlation_{held_out_building_name}.csv")
            target_corr.to_csv(target_corr_filename)
            print(f"    Feature-target correlation saved to {target_corr_filename}")

            fig_target_corr, ax_target_corr = plt.subplots(figsize=(12, max(8, len(target_corr)*0.35)), dpi=FIGURE_DPI)
            target_corr.plot(kind='barh', ax=ax_target_corr, color=sns.color_palette("coolwarm_r", len(target_corr)))
            ax_target_corr.set_title(f"Feature Correlation with Target (Train data for {held_out_building_name} fold)", fontsize=16)
            ax_target_corr.set_xlabel("Pearson Correlation", fontsize=12)
            plt.tight_layout(pad=1.5)
            target_corr_plot_filename = os.path.join(output_paths['feature_target_correlation_plots_dir'], f"feature_target_correlation_barchart_{held_out_building_name}.png")
            plt.savefig(target_corr_plot_filename)
            print(f"    Feature-target correlation plot saved to {target_corr_plot_filename}")
            plt.close(fig_target_corr)
        else:
            print("    Could not compute feature-target correlation: 'TARGET' column missing after merge.")

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
        'max_depth': ,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
    }

    print(f"  Training XGBoost for held-out building: {held_out_building_name}...")
    evals_result_xgb = {} 
    
    # --- TENSORBOARD INTEGRATION ---
    training_callbacks = []
    if TENSORBOARD_AVAILABLE:
        # Sanitize building name for directory
        safe_building_name = "".join(c if c.isalnum() else "_" for c in held_out_building_name)
        log_dir = os.path.join(output_paths['tensorboard_logs_dir'], f"fold_{safe_building_name}")
        os.makedirs(log_dir, exist_ok=True) # Ensure specific fold log dir exists
        
        # The TensorBoard callback needs a unique log_dir for each run/fold.
        # It will create subdirectories for train and eval within this log_dir.
        tensorboard_callback = TensorBoard(log_dir=log_dir, name=f"XGBoost_{safe_building_name}")
        training_callbacks.append(tensorboard_callback)
        print(f"    TensorBoard logging enabled. Logs will be saved to: {log_dir}")
    # --- END TENSORBOARD INTEGRATION ---

    model_xgb = xgb.train(
        params_xgb,
        dtrain_xgb,
        num_boost_round=1000, 
        evals=[(dtrain_xgb, 'train'), (dval_xgb, 'eval')],
        evals_result=evals_result_xgb, 
        early_stopping_rounds=100, 
        verbose_eval=100, 
        callbacks=training_callbacks # Add callbacks list here
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
    if not importance_df.empty: # Check before trying to operate on index/columns
        importance_df.index = pd.MultiIndex.from_tuples(importance_df.index, names=['importance_type', 'feature'])
        importance_df = importance_df.unstack(level='importance_type').fillna(0)
        if 'score' in importance_df.columns: 
            importance_df.columns = importance_df.columns.droplevel(0) 
    
    importance_filename = os.path.join(output_paths['importance_data_dir'], f"feature_importance_scores_{held_out_building_name}.csv")
    importance_df.to_csv(importance_filename)
    print(f"    Feature importance scores for fold saved to {importance_filename}")

    fold_result = {
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
    pred_filename = os.path.join(output_paths['predictions_data_dir'], f"predictions_{held_out_building_name}.csv")
    pred_df.to_csv(pred_filename, index=False)
    print(f"  Predictions saved to {pred_filename}")

    return fold_result
