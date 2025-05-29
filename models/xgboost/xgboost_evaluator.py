# models/xgboost/xgboost_evaluator.py

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg') # <<< ADD THIS LINE AT THE VERY TOP
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

FIGURE_DPI = 300 

def plot_overall_xgboost_results(cv_results, output_paths):
    """Plots overall MAPE comparison, actual vs. predicted for best model, 
       and feature importances for the best model. Saves plots."""
    if not cv_results:
        print("No XGBoost CV results to plot.")
        return

    print("\nPlotting Overall XGBoost LOOCV Results...")
    
    fig_mape, ax_mape = plt.subplots(figsize=(16, 8), dpi=FIGURE_DPI) 
    mapes_cv = {k: v['mape'] for k, v in cv_results.items() if v and 'mape' in v and pd.notna(v['mape'])} 
    
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
    plot_filename_mape = os.path.join(output_paths['plots_base_dir'], "xgboost_loocv_mape_comparison.png") 
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
        plot_filename_pred = os.path.join(output_paths['prediction_plots_dir'], f"xgboost_predictions_{best_bldg_cv}_best.png") 
        plt.savefig(plot_filename_pred)
        print(f"Best building prediction plot saved to {plot_filename_pred}")
        plt.close(fig_pred)

        if 'model' in result_best and hasattr(result_best['model'], 'get_score') and result_best.get('feature_names'): 
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
                        plot_filename_importance = os.path.join(output_paths['importance_plots_dir'], f"xgboost_feature_importance_{imp_type}_{best_bldg_cv}_best.png") 
                        plt.savefig(plot_filename_importance)
                        print(f"Feature importance plot ({imp_type}) saved to {plot_filename_importance}")
                    except Exception as e_imp:
                        print(f"Could not plot feature importance ({imp_type}) for {best_bldg_cv}: {e_imp}")
                    finally:
                        plt.close(fig_importance) 
    else:
        print("No best building to plot predictions for.")

def save_summary_stats(cv_results, output_paths):
    """Saves overall summary statistics from the cross-validation."""
    if not cv_results:
        print("No CV results to summarize.")
        return

    all_mapes = [res['mape'] for res in cv_results.values() if res and 'mape' in res and pd.notna(res['mape'])] 
    if all_mapes:
        avg_mape = np.mean(all_mapes)
        print(f"\nOverall Average MAPE from XGBoost LOOCV: {avg_mape:.2f}%")
        summary_stats = {
            "average_mape": avg_mape, 
            "individual_mapes": {k:v['mape'] for k,v in cv_results.items() if v and 'mape' in v and pd.notna(v['mape'])}
        }
        summary_filename = os.path.join(output_paths['data_dir'], "xgboost_loocv_summary.json") 
        try:
            with open(summary_filename, 'w') as f:
                json.dump(summary_stats, f, indent=4)
            print(f"Overall summary saved to {summary_filename}")
        except Exception as e:
            print(f"Error saving summary JSON: {e}")
    else:
        print("No valid MAPE scores to average for summary.")
