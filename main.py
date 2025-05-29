# main.py
# Orchestrates the Energy Consumption Forecasting pipeline with selective execution.

import data_cleaning # Imports the data_cleaning.py script
import eda
import baseline_models
# import gru_model # GRU model removed
import xgboost_hackathon
import os
import argparse # For command-line arguments

def run_pipeline(args):
    print("Starting Energy Consumption Forecasting Pipeline...\n")

    # Determine if any specific step was requested or if 'all' should run
    run_all_steps = args.all
    if not (args.clean or args.eda or args.baselines or args.xgboost):
        run_all_steps = True # Default to run all if no specific step is chosen

    # --- Step 1: Data Cleaning ---
    if args.clean or run_all_steps:
        print("--- Step 1: Data Cleaning ---")
        if not os.path.exists(data_cleaning.INPUT_XLSX):
            print(f"Error: Input Excel file '{data_cleaning.INPUT_XLSX}' for data cleaning not found.")
            print(f"Please ensure it's in the same directory as the scripts.")
            if not run_all_steps: return # Stop if only this step was requested and input is missing
        try:
            data_cleaning.main() 
            print("Data cleaning completed successfully.\n")
        except Exception as e:
            print(f"Error during data cleaning: {e}\n")
            if not run_all_steps: return 
            # If running all, we might want to stop the pipeline if cleaning fails critically
            print("Halting pipeline due to critical error in data cleaning.")
            return 
    
    # Check for cleaned files before proceeding to next steps if they are requested
    cleaned_electricity_path = os.path.join(data_cleaning.DATA_OUTPUT_DIR, data_cleaning.OUTPUT_ELECTRICITY_FILENAME)
    cleaned_weather_path = os.path.join(data_cleaning.DATA_OUTPUT_DIR, data_cleaning.OUTPUT_WEATHER_FILENAME)
    cleaned_areas_path = os.path.join(data_cleaning.DATA_OUTPUT_DIR, data_cleaning.OUTPUT_AREAS_FILENAME)
    
    required_cleaned_csvs = [
        cleaned_electricity_path,
        cleaned_weather_path,
        cleaned_areas_path
    ]

    # --- Step 2: Exploratory Data Analysis ---
    if args.eda or run_all_steps:
        print("--- Step 2: Exploratory Data Analysis ---")
        missing_cleaned_for_eda = [f for f in [cleaned_electricity_path, cleaned_areas_path] if not os.path.exists(f)]
        if missing_cleaned_for_eda:
            print(f"Error: EDA requires cleaned data. Missing: {', '.join(missing_cleaned_for_eda)}")
            if not run_all_steps: return
            print("Skipping EDA due to missing cleaned files.")
        else:
            try:
                eda.main() 
                print("EDA completed successfully.\n")
            except Exception as e:
                print(f"Error during EDA: {e}\n")
                if not run_all_steps: return

    # --- Step 3: Baseline Models ---
    if args.baselines or run_all_steps:
        print("--- Step 3: Baseline Models ---")
        missing_cleaned_for_baselines = [f for f in [cleaned_electricity_path] if not os.path.exists(f)] # Baselines might only need electricity
        if missing_cleaned_for_baselines:
            print(f"Error: Baseline models require cleaned electricity data. Missing: {', '.join(missing_cleaned_for_baselines)}")
            if not run_all_steps: return
            print("Skipping Baseline Models due to missing cleaned files.")
        else:
            try:
                baseline_models.main() 
                print("Baseline models evaluation completed successfully.\n")
            except Exception as e:
                print(f"Error during baseline model evaluation: {e}\n")
                if not run_all_steps: return
    
    # --- Step 4: Advanced Model - XGBoost Hackathon Simulation (LOOCV) ---
    if args.xgboost or run_all_steps:
        print("--- Step 4: Advanced Model - XGBoost Hackathon Simulation (LOOCV) ---")
        missing_cleaned_for_xgb = [f for f in required_cleaned_csvs if not os.path.exists(f)] # XGBoost needs all
        if missing_cleaned_for_xgb:
            print(f"Error: XGBoost simulation requires all cleaned data. Missing: {', '.join(missing_cleaned_for_xgb)}")
            if not run_all_steps: return
            print("Skipping XGBoost simulation due to missing cleaned files.")
        else:
            # Confirm before running XGBoost if it's part of "all" and not explicitly requested
            run_xgb_confirmed = args.xgboost 
            if run_all_steps and not args.xgboost: # If running all and xgboost wasn't specifically asked for
                 confirm_xgb = input("XGBoost simulation can be time-consuming. Run it? (yes/no): ").strip().lower()
                 if confirm_xgb == 'yes':
                     run_xgb_confirmed = True
                 else:
                     print("Skipping XGBoost Hackathon simulation as per user input.\n")
            
            if run_xgb_confirmed:
                try:
                    xgboost_hackathon.main() 
                    print("XGBoost Hackathon simulation completed.\n")
                except ImportError as ie:
                    print(f"ImportError: {ie}. XGBoost or Matplotlib (or their dependencies) not found. Skipping XGBoost simulation.\n")
                except Exception as e:
                    print(f"Error during XGBoost Hackathon simulation: {e}\n")
                    if not run_all_steps: return
            elif run_all_steps and not args.xgboost: # If it was skipped as part of 'all'
                pass # Already printed skipping message
            
    print("--- Energy Consumption Forecasting Pipeline Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run parts of the Energy Consumption Forecasting pipeline.")
    parser.add_argument('--clean', action='store_true', help="Run only the data cleaning step.")
    parser.add_argument('--eda', action='store_true', help="Run only the Exploratory Data Analysis step.")
    parser.add_argument('--baselines', action='store_true', help="Run only the baseline models evaluation step.")
    parser.add_argument('--xgboost', action='store_true', help="Run only the XGBoost hackathon simulation step.")
    parser.add_argument('--all', action='store_true', help="Run all steps of the pipeline (default if no other step is specified).")
    
    # You can add an argument to skip confirmations if needed, e.g., --yes-to-all
    # parser.add_argument('-y', '--yes', action='store_true', help="Automatically answer yes to confirmations.")

    parsed_args = parser.parse_args()
    run_pipeline(parsed_args)
