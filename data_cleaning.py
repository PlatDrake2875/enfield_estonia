# data_cleaning.py (Main Orchestrator)
# Uses preprocessor classes to clean data and save to CSVs.

import pandas as pd 
import os

# Import the cleaner classes from the data_preprocessors package
from data_preprocessors.electricity_cleaner import ElectricityCleaner
from data_preprocessors.weather_cleaner import WeatherCleaner
from data_preprocessors.areas_cleaner import AreasCleaner

# Define the names of the input Excel file
INPUT_XLSX = 'Buildings_el.xlsx' # Assumed to be in the same directory as this script

# --- THESE CONSTANTS MUST BE DEFINED HERE ---
# Define the output directory for cleaned CSV files
DATA_OUTPUT_DIR = 'data' # Will be created if it doesn't exist

# Define the names for the output CSV files (filenames only)
OUTPUT_ELECTRICITY_FILENAME = 'cleaned_electricity.csv'
OUTPUT_WEATHER_FILENAME = 'cleaned_weather.csv'
OUTPUT_AREAS_FILENAME = 'cleaned_areas.csv'
# --- END CONSTANTS DEFINITION ---

# Sheet names (as expected in the input Excel file)
ELECTRICITY_SHEET = 'Electricity kWh'
WEATHER_SHEET = 'Weather archive'
AREAS_SHEET = 'Areas'

def create_output_data_dir():
    """Creates the output data directory if it doesn't exist."""
    if not os.path.exists(DATA_OUTPUT_DIR):
        os.makedirs(DATA_OUTPUT_DIR)
        print(f"Created output directory: '{DATA_OUTPUT_DIR}'")
    else:
        print(f"Output directory '{DATA_OUTPUT_DIR}' already exists or was just created.")

def save_cleaned_data_to_csv(elec_df, weather_df, areas_df):
    """Saves the cleaned dataframes to separate CSV files in the DATA_OUTPUT_DIR."""
    create_output_data_dir() 

    if elec_df is not None:
        output_path = os.path.join(DATA_OUTPUT_DIR, OUTPUT_ELECTRICITY_FILENAME)
        try:
            elec_df.to_csv(output_path, index=False)
            print(f"✅ Cleaned electricity data saved to '{output_path}'")
        except Exception as e:
            print(f"Error saving cleaned electricity data to CSV: {e}")
    else:
        print(f"Electricity dataframe is None. Skipping save for {OUTPUT_ELECTRICITY_FILENAME}.")

    if weather_df is not None:
        output_path = os.path.join(DATA_OUTPUT_DIR, OUTPUT_WEATHER_FILENAME)
        try:
            weather_df_to_save = weather_df.copy()
            if 'Datetime' in weather_df_to_save.columns and 'time_iso' not in weather_df_to_save.columns:
                 weather_df_to_save['time_iso'] = weather_df_to_save['Datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')
            if 'time_iso' in weather_df_to_save.columns and 'Datetime' in weather_df_to_save.columns:
                weather_df_to_save = weather_df_to_save.drop(columns=['Datetime'], errors='ignore')
            if 'time_iso' in weather_df_to_save.columns:
                cols = ['time_iso'] + [col for col in weather_df_to_save.columns if col != 'time_iso']
                cols = [c for c in cols if c in weather_df_to_save.columns]
                weather_df_to_save = weather_df_to_save[cols]
            
            weather_df_to_save.to_csv(output_path, index=False)
            print(f"✅ Cleaned weather data saved to '{output_path}'")
        except Exception as e:
            print(f"Error saving cleaned weather data to CSV: {e}")
    else:
        print(f"Weather dataframe is None. Skipping save for {OUTPUT_WEATHER_FILENAME}.")

    if areas_df is not None:
        output_path = os.path.join(DATA_OUTPUT_DIR, OUTPUT_AREAS_FILENAME)
        try:
            areas_df.to_csv(output_path, index=False)
            print(f"✅ Cleaned areas data saved to '{output_path}'")
        except Exception as e:
            print(f"Error saving cleaned areas data to CSV: {e}")
    else:
        print(f"Areas dataframe is None. Skipping save for {OUTPUT_AREAS_FILENAME}.")


def main():
    """Main function to run the data cleaning process using preprocessor classes."""
    if not os.path.exists(INPUT_XLSX):
        print(f"Error: Input Excel file '{INPUT_XLSX}' not found.")
        print(f"Please ensure the file '{INPUT_XLSX}' is in the same directory as this script.")
        return

    # Initialize cleaners
    elec_cleaner = ElectricityCleaner(INPUT_XLSX, ELECTRICITY_SHEET)
    weather_cleaner = WeatherCleaner(INPUT_XLSX, WEATHER_SHEET)
    areas_cleaner = AreasCleaner(INPUT_XLSX, AREAS_SHEET)

    # Process data
    elec_cleaned = elec_cleaner.process()
    weather_cleaned = weather_cleaner.process()
    areas_cleaned = areas_cleaner.process()
    
    if elec_cleaned is None or weather_cleaned is None or areas_cleaned is None:
        print("One or more data cleaning steps failed. Output CSV files may be incomplete or not generated.")
    
    save_cleaned_data_to_csv(elec_cleaned, weather_cleaned, areas_cleaned)
    print("Data cleaning pipeline finished.")

if __name__ == '__main__':
    main()
