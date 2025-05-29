# data_preprocessors/areas_cleaner.py

import pandas as pd
import numpy as np
import os

class AreasCleaner:
    def __init__(self, input_xlsx_path, sheet_name):
        self.input_xlsx_path = input_xlsx_path
        self.sheet_name = sheet_name
        print(f"Initialized AreasCleaner for sheet: '{self.sheet_name}'")

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
        """Reads the areas data from the specified Excel sheet."""
        if not os.path.exists(self.input_xlsx_path):
            print(f"Error: Input Excel file '{self.input_xlsx_path}' not found.")
            return None
        try:
            areas_df = pd.read_excel(self.input_xlsx_path, sheet_name=self.sheet_name)
            print(f"  Successfully read sheet: '{self.sheet_name}'")
            return areas_df
        except FileNotFoundError:
            print(f"Error: Input Excel file '{self.input_xlsx_path}' not found (checked again).")
            return None
        except ValueError as ve:
            print(f"Error reading sheet '{self.sheet_name}' from '{self.input_xlsx_path}': {ve}")
            return None
        except Exception as e:
            print(f"Error reading areas sheet '{self.sheet_name}' from '{self.input_xlsx_path}': {e}")
            return None

    def _normalize_and_clean(self, areas_df):
        """Converts 'Area [m2]' to numeric and fills NaNs."""
        if 'Area [m2]' not in areas_df.columns:
            potential_area_col = next((col for col in areas_df.columns if 'area' in str(col).lower()), None)
            if potential_area_col:
                print(f"  Found potential area column: '{potential_area_col}'. Using this column and renaming to 'Area [m2]'.")
                areas_df.rename(columns={potential_area_col: 'Area [m2]'}, inplace=True)
            else:
                print(f"  Error: 'Area [m2]' column not found and no alternative found.")
                return None # Indicate failure

        areas_df.loc[:, 'Area [m2]'] = pd.to_numeric(areas_df['Area [m2]'], errors='coerce')
        areas_df.ffill(inplace=True)
        areas_df.bfill(inplace=True)
        return areas_df

    def process(self):
        """Main method to clean the areas data."""
        print(f"Processing areas data from sheet: '{self.sheet_name}'...")
        areas_df = self._read_data()
        if areas_df is None:
            return None
        
        areas_df = self._normalize_and_clean(areas_df)
        if areas_df is None:
            return None
            
        areas_df = AreasCleaner._bump_zeros(areas_df)
        print(f"Areas data cleaning for sheet '{self.sheet_name}' complete.")
        return areas_df

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    test_input_file = os.path.join(project_root, 'Buildings_el.xlsx')
    
    if os.path.exists(test_input_file):
        cleaner = AreasCleaner(test_input_file, 'Areas')
        cleaned_df = cleaner.process()
        if cleaned_df is not None:
            print("\nCleaned Areas Data Sample (from direct test):")
            print(cleaned_df.head())
    else:
        print(f"Test input file not found: {test_input_file}")
