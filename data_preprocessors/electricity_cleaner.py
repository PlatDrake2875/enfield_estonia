# data_preprocessors/electricity_cleaner.py

import pandas as pd
import numpy as np
import os

class ElectricityCleaner:
    def __init__(self, input_xlsx_path, sheet_name):
        self.input_xlsx_path = input_xlsx_path
        self.sheet_name = sheet_name
        print(f"Initialized ElectricityCleaner for sheet: '{self.sheet_name}'")

    @staticmethod
    def _bump_zeros(df):
        """Replace exact zeros in numeric columns with a small positive value."""
        if df is None: 
            return None
        num_cols = df.select_dtypes(include=[np.number]).columns
        df_copy = df.copy() 
        df_copy[num_cols] = df_copy[num_cols].replace(0, 0.001)
        return df_copy

    def _read_data(self):
        """Reads the electricity data from the specified Excel sheet."""
        if not os.path.exists(self.input_xlsx_path):
            print(f"Error: Input Excel file '{self.input_xlsx_path}' not found.")
            return None
        try:
            elec_df = pd.read_excel(self.input_xlsx_path, sheet_name=self.sheet_name, header=[0,1])
            print(f"  Successfully read sheet: '{self.sheet_name}'")
            return elec_df
        except FileNotFoundError:
            print(f"Error: Input Excel file '{self.input_xlsx_path}' not found (checked again).")
            return None
        except ValueError as ve: 
            print(f"Error reading sheet '{self.sheet_name}' from '{self.input_xlsx_path}': {ve}")
            return None
        except Exception as e:
            print(f"Error reading electricity sheet '{self.sheet_name}' from '{self.input_xlsx_path}': {e}")
            return None

    def _process_columns(self, elec_df):
        """Processes column names and identifies the Timestamp column."""
        new_columns = []
        for col_tuple in elec_df.columns:
            if isinstance(col_tuple, tuple):
                lvl1_val = str(col_tuple[1]) 
                if 'timestamp' in lvl1_val.lower(): 
                    new_columns.append('Timestamp')
                else:
                    new_columns.append(lvl1_val.strip())
            else: 
                lvl1_val = str(col_tuple)
                if 'timestamp' in lvl1_val.lower(): 
                    new_columns.append('Timestamp')
                else:
                    new_columns.append(lvl1_val.strip())
        elec_df.columns = new_columns
        
        if 'Timestamp' not in elec_df.columns:
            print(f"  Error: 'Timestamp' column not found after processing headers.")
            potential_ts_col = next((col for col in elec_df.columns if 'time' in str(col).lower() or 'date' in str(col).lower()), None)
            if potential_ts_col:
                print(f"  Found potential timestamp column: '{potential_ts_col}'. Using this column and renaming to 'Timestamp'.")
                elec_df.rename(columns={potential_ts_col: 'Timestamp'}, inplace=True)
            else:
                print("  Could not identify a timestamp column. Processing might fail.")
                return None 
        return elec_df

    def _normalize_and_clean(self, elec_df_in): # Operate on a copy
        """Normalizes timestamps, converts to numeric, applies rolling mean, and fills NaNs."""
        elec_df = elec_df_in.copy() # Work on a copy to avoid warnings on original df passed

        if 'Timestamp' not in elec_df.columns: 
             print("  Error: Timestamp column missing before normalization.")
             return None

        elec_df.loc[:, 'Timestamp'] = pd.to_datetime(elec_df['Timestamp'], errors='coerce')

        if elec_df['Timestamp'].isnull().all():
            print(f"  Error: All 'Timestamp' values are NaT after conversion. Please check the date format.")
            return None

        for col in elec_df.columns.drop('Timestamp', errors='ignore'): 
            if col == 'time_iso': continue 
            # Convert to numeric, which will likely make it float if it has NaNs or decimals
            numeric_series = pd.to_numeric(elec_df[col], errors='coerce')
            # Apply rolling mean (this will be float)
            rolled_series = numeric_series.rolling(3, center=True, min_periods=1).mean()
            # Assign back. If original column was int, this might change its dtype to float.
            elec_df.loc[:, col] = rolled_series

        elec_df.loc[:, 'time_iso'] = elec_df['Timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        cols_to_keep = ['time_iso'] + [c for c in elec_df.columns if c not in ['Timestamp', 'time_iso']]
        # Ensure this is a new DataFrame, not a view, to avoid SettingWithCopyWarning on ffill/bfill
        elec_df_processed = elec_df[cols_to_keep].copy() 
        
        elec_df_processed.ffill(inplace=True)
        elec_df_processed.bfill(inplace=True) 
        return elec_df_processed

    def process(self):
        """Main method to clean the electricity data."""
        print(f"Processing electricity data from sheet: '{self.sheet_name}'...")
        elec_df = self._read_data()
        if elec_df is None:
            return None
        
        elec_df_processed_cols = self._process_columns(elec_df) # This modifies elec_df inplace for columns
        if elec_df_processed_cols is None: # if _process_columns returned None due to missing Timestamp
            return None
        
        # _normalize_and_clean now operates on a copy of the input
        elec_df_cleaned = self._normalize_and_clean(elec_df_processed_cols) 
        if elec_df_cleaned is None:
            return None

        elec_df_final = ElectricityCleaner._bump_zeros(elec_df_cleaned)
        print(f"Electricity data cleaning for sheet '{self.sheet_name}' complete.")
        return elec_df_final

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) 
    test_input_file = os.path.join(project_root, 'Buildings_el.xlsx')
    
    if os.path.exists(test_input_file):
        cleaner = ElectricityCleaner(test_input_file, 'Electricity kWh')
        cleaned_df = cleaner.process()
        if cleaned_df is not None:
            print("\nCleaned Electricity Data Sample (from direct test):")
            print(cleaned_df.head())
    else:
        print(f"Test input file not found: {test_input_file}")

