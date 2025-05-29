# data_cleaning.py
# Energy Consumption Forecasting
# Section 2: Data Loading & Cleaning

# This script will:
# 1. Import raw data from an Excel file (`Buildings_el.xlsx`) with multiple sheets.
# 2. Clean and normalize timestamps and numeric values for each sheet.
# 3. Export cleaned data to separate CSV files in a 'data' subdirectory.

import pandas as pd
import numpy as np
import re
import os

# Define the names of the input Excel file
INPUT_XLSX = 'Buildings_el.xlsx'

# Define the output directory for cleaned CSV files
DATA_OUTPUT_DIR = 'data'

# Define the names for the output CSV files (filenames only)
OUTPUT_ELECTRICITY_FILENAME = 'cleaned_electricity.csv'
OUTPUT_WEATHER_FILENAME = 'cleaned_weather.csv'
OUTPUT_AREAS_FILENAME = 'cleaned_areas.csv'

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


# bump_zeros function
def bump_zeros(df):
    """Replace exact zeros in numeric columns with a small positive value."""
    if df is None: # Add check for None df
        return None
    num_cols = df.select_dtypes(include=[np.number]).columns
    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    df_copy[num_cols] = df_copy[num_cols].replace(0, 0.001)
    return df_copy

def clean_electricity_data(input_xlsx_path, sheet_name):
    """Cleans the electricity consumption data from the specified Excel sheet."""
    print(f"Cleaning electricity data from sheet: '{sheet_name}'...")
    if not os.path.exists(input_xlsx_path):
        print(f"Error: Input Excel file '{input_xlsx_path}' not found.")
        return None
    try:
        elec = pd.read_excel(input_xlsx_path, sheet_name=sheet_name, header=[0,1])
    except FileNotFoundError:
        print(f"Error: Input Excel file '{input_xlsx_path}' not found (checked again).")
        return None
    except ValueError as ve: 
        print(f"Error reading sheet '{sheet_name}' from '{input_xlsx_path}': {ve}")
        return None
    except Exception as e:
        print(f"Error reading electricity sheet '{sheet_name}' from '{input_xlsx_path}': {e}")
        return None

    new_columns = []
    for col_tuple in elec.columns:
        if isinstance(col_tuple, tuple):
            lvl1_val = str(col_tuple[1]) 
            if 'timestamp' in lvl1_val.lower(): # Check if 'timestamp' (or similar) is in the second part
                new_columns.append('Timestamp')
            else:
                new_columns.append(lvl1_val.strip())
        else: 
            lvl1_val = str(col_tuple)
            if 'timestamp' in lvl1_val.lower(): # Check if 'timestamp' is in the column name itself
                new_columns.append('Timestamp')
            else:
                new_columns.append(lvl1_val.strip())
                
    elec.columns = new_columns
    
    if 'Timestamp' not in elec.columns:
        print(f"Error: 'Timestamp' column not found after processing headers in sheet '{sheet_name}'.")
        potential_ts_col = next((col for col in elec.columns if 'time' in str(col).lower() or 'date' in str(col).lower()), None)
        if potential_ts_col:
            print(f"Found potential timestamp column: '{potential_ts_col}'. Using this column and renaming to 'Timestamp'.")
            elec.rename(columns={potential_ts_col: 'Timestamp'}, inplace=True)
        else:
            print("Could not identify a timestamp column. Electricity data cleaning might fail or be incorrect.")
            return None

    elec['Timestamp'] = pd.to_datetime(elec['Timestamp'], errors='coerce')

    if elec['Timestamp'].isnull().all():
        print(f"Error: All 'Timestamp' values are NaT after conversion in sheet '{sheet_name}'. Please check the date format.")
        return None

    for col in elec.columns.drop('Timestamp', errors='ignore'): 
        if col == 'time_iso': continue 
        elec[col] = pd.to_numeric(elec[col], errors='coerce') \
                     .rolling(3, center=True, min_periods=1).mean()

    elec['time_iso'] = elec['Timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    cols_to_keep = ['time_iso'] + [c for c in elec.columns if c not in ['Timestamp', 'time_iso']]
    elec = elec[cols_to_keep]
    
    elec.ffill(inplace=True)
    elec.bfill(inplace=True) 
    elec = bump_zeros(elec)
    print(f"Electricity data cleaning from sheet '{sheet_name}' complete.")
    return elec

def clean_weather_data(input_xlsx_path, sheet_name):
    """Cleans the weather archive data from the specified Excel sheet."""
    print(f"Cleaning weather data from sheet: '{sheet_name}'...")
    if not os.path.exists(input_xlsx_path):
        print(f"Error: Input Excel file '{input_xlsx_path}' not found.")
        return None
    try:
        weather = pd.read_excel(input_xlsx_path, sheet_name=sheet_name, skiprows=1, header=0) 
    except FileNotFoundError:
        print(f"Error: Input Excel file '{input_xlsx_path}' not found (checked again).")
        return None
    except ValueError as ve:
        print(f"Error reading sheet '{sheet_name}' from '{input_xlsx_path}': {ve}")
        return None
    except Exception as e:
        print(f"Error reading weather sheet '{sheet_name}' from '{input_xlsx_path}': {e}")
        return None

    drops = []
    for name in ['total cloud cover','Max gust value','weather phenomena','atm pressure to sea level']:
        cols_to_drop_by_name = [c for c in weather.columns if isinstance(c, str) and c.strip().lower()==name]
        drops.extend(cols_to_drop_by_name)
    weather.drop(columns=drops, errors='ignore', inplace=True)

    if not weather.empty and len(weather.columns) > 0:
        orig_ts_col_name = weather.columns[0] 
        weather['time_iso'] = pd.to_datetime(
            weather[orig_ts_col_name], format='%d.%m.%Y %H:%M', dayfirst=True, errors='coerce'
        ).dt.strftime('%Y-%m-%dT%H:%M:%S') 
        weather.drop(columns=[orig_ts_col_name], inplace=True)
    else:
        print(f"Warning: Weather data from sheet '{sheet_name}' is empty or has no columns to process for timestamp.")
        return weather 

    def map_wind_norm(txt):
        if pd.isna(txt): return np.nan
        piece = re.sub(r'[^A-Za-z\\ -]','',str(txt).split('from the')[-1]).lower().strip()
        deg = {'north':0,'northeast':45,'east':90,'southeast':135,'south':180,'southwest':225,'west':270,'northwest':315}
        vals = [deg[t] for t in re.split('[\\\\-\\\\s]+',piece) if t in deg] 
        return np.mean(vals)/360 if vals else np.nan

    if 'Mean wind direction' in weather.columns:
        weather['Mean wind direction'] = weather['Mean wind direction'].apply(map_wind_norm)

    vis_col_name = next((c for c in weather.columns if isinstance(c, str) and c.lower()=='visibility'), None)
    if vis_col_name:
        weather[vis_col_name] = (weather[vis_col_name].astype(str)
                         .str.extract(r'(\d+\.?\d*)')[0].astype(float))
        vmin, vmax = weather[vis_col_name].min(), weather[vis_col_name].max()
        if pd.notna(vmin) and pd.notna(vmax) and vmax > vmin: 
            weather[vis_col_name] = (weather[vis_col_name] - vmin)/(vmax-vmin)
        elif pd.notna(vmin) and pd.notna(vmax) and vmax == vmin: 
             weather[vis_col_name] = 0.0 if vmin != 0 else 0.0 
        else: 
            weather[vis_col_name] = np.nan 

    cols_to_exclude_from_numeric = ['time_iso']
    if 'Mean wind direction' in weather.columns:
        cols_to_exclude_from_numeric.append('Mean wind direction')
    if vis_col_name:
        cols_to_exclude_from_numeric.append(vis_col_name)
    
    numeric_cols_to_convert = weather.columns.difference(cols_to_exclude_from_numeric)
    for c in numeric_cols_to_convert:
        weather[c] = pd.to_numeric(weather[c], errors='coerce')
        
    weather = weather.sort_values('time_iso').reset_index(drop=True)
    cols_order = ['time_iso'] + [c for c in weather if c != 'time_iso']
    weather = weather[cols_order]
    
    weather.ffill(inplace=True)
    weather.bfill(inplace=True)
    weather = bump_zeros(weather)
    print(f"Weather data cleaning from sheet '{sheet_name}' complete.")
    return weather

def clean_areas_data(input_xlsx_path, sheet_name):
    """Cleans the areas data from the specified Excel sheet."""
    print(f"Cleaning areas data from sheet: '{sheet_name}'...")
    if not os.path.exists(input_xlsx_path):
        print(f"Error: Input Excel file '{input_xlsx_path}' not found.")
        return None
    try:
        areas = pd.read_excel(input_xlsx_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"Error: Input Excel file '{input_xlsx_path}' not found (checked again).")
        return None
    except ValueError as ve:
        print(f"Error reading sheet '{sheet_name}' from '{input_xlsx_path}': {ve}")
        return None
    except Exception as e:
        print(f"Error reading areas sheet '{sheet_name}' from '{input_xlsx_path}': {e}")
        return None
        
    if 'Area [m2]' not in areas.columns:
        potential_area_col = next((col for col in areas.columns if 'area' in str(col).lower()), None)
        if potential_area_col:
            print(f"Found potential area column: '{potential_area_col}'. Using this column and renaming to 'Area [m2]'.")
            areas.rename(columns={potential_area_col: 'Area [m2]'}, inplace=True)
        else:
            print(f"Error: 'Area [m2]' column not found in sheet '{sheet_name}' and no alternative found.")
            return None 

    areas['Area [m2]'] = pd.to_numeric(areas['Area [m2]'], errors='coerce')
    areas.ffill(inplace=True)
    areas.bfill(inplace=True)
    areas = bump_zeros(areas)
    print(f"Areas data cleaning from sheet '{sheet_name}' complete.")
    return areas

def save_cleaned_data_to_csv(elec_df, weather_df, areas_df):
    """Saves the cleaned dataframes to separate CSV files in the DATA_OUTPUT_DIR."""
    create_output_data_dir() # Ensure directory exists

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
            weather_df.to_csv(output_path, index=False)
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
    """Main function to run the data cleaning process."""
    if not os.path.exists(INPUT_XLSX):
        print(f"Error: Input Excel file '{INPUT_XLSX}' not found.")
        print(f"Please ensure the file '{INPUT_XLSX}' is in the same directory as this script.")
        return

    elec_cleaned = clean_electricity_data(INPUT_XLSX, ELECTRICITY_SHEET)
    weather_cleaned = clean_weather_data(INPUT_XLSX, WEATHER_SHEET)
    areas_cleaned = clean_areas_data(INPUT_XLSX, AREAS_SHEET)
    
    if elec_cleaned is None or weather_cleaned is None or areas_cleaned is None:
        print("One or more data cleaning steps failed. Output CSV files may be incomplete or not generated.")
    
    save_cleaned_data_to_csv(elec_cleaned, weather_cleaned, areas_cleaned)

if __name__ == '__main__':
    main()
