# main.py
# Orchestrates the Energy Consumption Forecasting pipeline with selective execution.

import data_cleaning 
import eda
import baseline_models
# import gru_model # GRU model removed
# import xgboost_hackathon # Replaced by models.xgboost_pipeline
import os
import argparse 

# --- NEW IMPORT ---
from models import xgboost_pipeline # Import the new XGBoost pipeline module

def run_pipeline(args):
    print("Starting Energy Consumption Forecasting Pipeline...\n")

    run_all_steps = args.all
    if not (args.clean or args.eda or args.baselines or args.xgboost):
        run_all_steps = True 

    # --- Step 1: Data Cleaning ---
    if args.clean or run_all_steps:
        print("--- Step 1: Data Cleaning ---")
        if not os.path.exists(data_cleaning.INPUT_XLSX): # INPUT_XLSX is defined in data_cleaning.py
            print(f"Error: Input Excel file '{data_cleaning.INPUT_XLSX}' for data cleaning not found.")
            print(f"Please ensure it's in the same directory as the scripts.")
            if not run_all_steps: return 
        try:
            data_cleaning.main() 
            print("Data cleaning completed successfully.\n")
        except Exception as e:
            print(f"Error during data cleaning: {e}\n")
            if not run_all_steps: return 
            print("Halting pipeline due to critical error in data cleaning.")
            return 
    
    # Construct paths to cleaned CSVs using constants from data_cleaning.py
    # These paths are relative to where main.py is run, assuming 'data' is a subdir.
    cleaned_electricity_path = os.path.join(data_cleaning.DATA_OUTPUT_DIR, data_cleaning.OUTPUT_ELECTRICITY_FILENAME)
    cleaned_weather_path = os.path.join(data_cleaning.DATA_OUTPUT_DIR, data_cleaning.OUTPUT_WEATHER_FILENAME)
    cleaned_areas_path = os.path.join(data_cleaning.DATA_OUTPUT_DIR, data_cleaning.OUTPUT_AREAS_FILENAME)
    
    required_cleaned_csvs_for_downstream = [
        cleaned_electricity_path,
        cleaned_weather_path, # XGBoost and potentially baselines might need it
        cleaned_areas_path
    ]

    # --- Step 2: Exploratory Data Analysis ---
    if args.eda or run_all_steps:
        print("--- Step 2: Exploratory Data Analysis ---")
        # EDA needs electricity and areas CSVs from the 'data' folder
        missing_for_eda = [f for f in [cleaned_electricity_path, cleaned_areas_path] if not os.path.exists(f)]
        if missing_for_eda:
            print(f"Error: EDA requires cleaned data. Missing: {', '.join(missing_for_eda)}")
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
        # Baselines typically need at least cleaned_electricity.csv
        # Update this check if baseline_models.py has different dependencies
        missing_for_baselines = [f for f in [cleaned_electricity_path] if not os.path.exists(f)] 
        if missing_for_baselines:
            print(f"Error: Baseline models require cleaned electricity data. Missing: {', '.join(missing_for_baselines)}")
            if not run_all_steps: return
            print("Skipping Baseline Models due to missing cleaned files.")
        else:
            try:
                baseline_models.main() # Ensure baseline_models.py reads from data/cleaned_electricity.csv etc.
                print("Baseline models evaluation completed successfully.\n")
            except Exception as e:
                print(f"Error during baseline model evaluation: {e}\n")
                if not run_all_steps: return
    
    # --- Step 4: Advanced Model - XGBoost Hackathon Simulation (LOOCV) ---
    if args.xgboost or run_all_steps:
        print("--- Step 4: Advanced Model - XGBoost Pipeline (LOOCV) ---")
        # XGBoost pipeline's data_preparer will look for files in 'data/' relative to project root
        # So, we just need to ensure data_cleaning ran successfully.
        # The actual check for individual CSVs is now inside models.xgboost_data_preparer.prepare_data()
        
        # A preliminary check here to ensure the 'data' dir itself exists if cleaning wasn't run in this session
        if not os.path.exists(data_cleaning.DATA_OUTPUT_DIR) and not args.clean : # If data dir doesn't exist and we didn't just run clean
             print(f"Error: XGBoost requires cleaned data in '{data_cleaning.DATA_OUTPUT_DIR}'. This directory is missing.")
             print("Please run the --clean step first or ensure the directory and its files exist.")
             if not run_all_steps: return
             print("Skipping XGBoost pipeline.")

        else: # Proceed if data dir exists or if cleaning was part of the run
            run_xgb_confirmed = args.xgboost 
            if run_all_steps and not args.xgboost: 
                 confirm_xgb = input("XGBoost pipeline can be time-consuming. Run it? (yes/no): ").strip().lower()
                 if confirm_xgb == 'yes':
                     run_xgb_confirmed = True
                 else:
                     print("Skipping XGBoost pipeline as per user input.\n")
            
            if run_xgb_confirmed:
                try:
                    # --- UPDATED CALL ---
                    xgboost_pipeline.main() 
                    print("XGBoost pipeline completed.\n")
                except ImportError as ie:
                    print(f"ImportError: {ie}. Could not import XGBoost pipeline modules. Check structure and __init__.py in 'models'.\n")
                except FileNotFoundError as fnfe: # Catch FileNotFoundError from prepare_data if it's re-raised
                    print(f"FileNotFoundError during XGBoost pipeline: {fnfe}. Ensure cleaned data files exist in 'data/'.\n")
                except Exception as e:
                    print(f"Error during XGBoost pipeline: {e}\n")
                    if not run_all_steps: return
            elif run_all_steps and not args.xgboost: 
                pass 
            
    print("--- Energy Consumption Forecasting Pipeline Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run parts of the Energy Consumption Forecasting pipeline.")
    parser.add_argument('--clean', action='store_true', help="Run only the data cleaning step.")
    parser.add_argument('--eda', action='store_true', help="Run only the Exploratory Data Analysis step.")
    parser.add_argument('--baselines', action='store_true', help="Run only the baseline models evaluation step.")
    parser.add_argument('--xgboost', action='store_true', help="Run only the XGBoost pipeline step.")
    parser.add_argument('--all', action='store_true', help="Run all steps of the pipeline (default if no other step is specified).")
    
    parsed_args = parser.parse_args()
    run_pipeline(parsed_args)
