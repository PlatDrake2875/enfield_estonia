# data_preprocessors/weather_cleaner.py

import pandas as pd
import numpy as np
import re
import os

class WeatherCleaner:
    def __init__(self, input_xlsx_path, sheet_name):
        self.input_xlsx_path = input_xlsx_path
        self.sheet_name = sheet_name
        print(f"Initialized WeatherCleaner for sheet: '{self.sheet_name}'")

    @staticmethod
    def _bump_zeros(df):
        if df is None: return None
        num_cols = df.select_dtypes(include=[np.number]).columns
        df_copy = df.copy() 
        df_copy[num_cols] = df_copy[num_cols].replace(0, 0.001)
        return df_copy

    def _read_data(self):
        if not os.path.exists(self.input_xlsx_path):
            print(f"Error: Input Excel file '{self.input_xlsx_path}' not found.")
            return None
        try:
            weather_df = pd.read_excel(self.input_xlsx_path, sheet_name=self.sheet_name, skiprows=1, header=0) 
            print(f"  Successfully read sheet: '{self.sheet_name}'")
            return weather_df
        except Exception as e:
            print(f"Error reading weather sheet '{self.sheet_name}' from '{self.input_xlsx_path}': {e}")
            return None

    def _weather_drop_unwanted_columns(self, weather_df):
        """Drops specified unwanted columns from the weather DataFrame."""
        print("  Dropping unwanted weather columns...")
        cols_to_drop = []
        
        # Columns to identify and drop (case-insensitive substring match)
        columns_to_remove_keywords = [
            "max gust value", 
            "dewpoint temperature", # Added
            "atm pressure to sea level", # Added (will catch variants)
            "atmospheric pressure reduced to mean sea level" # Added variant
        ]

        for col_name_in_df in weather_df.columns:
            if isinstance(col_name_in_df, str):
                col_name_lower = col_name_in_df.strip().lower()
                for keyword in columns_to_remove_keywords:
                    if keyword.lower() in col_name_lower:
                        if col_name_in_df not in cols_to_drop: # Avoid duplicates
                            cols_to_drop.append(col_name_in_df)
                        break # Found a match for this column, move to next column in df
        
        weather_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
        print(f"    Dropped columns (if they existed): {cols_to_drop}")
        return weather_df

    def _weather_process_timestamp(self, weather_df_in):
        print("  Processing weather timestamp...")
        if weather_df_in.empty or len(weather_df_in.columns) == 0:
            print("    Warning: Weather data is empty, skipping timestamp processing.")
            return weather_df_in.copy()
        
        weather_df = weather_df_in.copy()
        orig_ts_col_name = weather_df.columns[0] 
        temp_datetime_col = pd.to_datetime(
            weather_df[orig_ts_col_name], format='%d.%m.%Y %H:%M', dayfirst=True, errors='coerce'
        )
        valid_ts_mask = temp_datetime_col.notna()
        weather_df = weather_df.loc[valid_ts_mask].copy()
        
        if weather_df.empty:
            print(f"    Warning: All rows dropped due to unparseable original timestamps.")
            return weather_df
            
        weather_df.loc[:, 'time_iso'] = temp_datetime_col[valid_ts_mask].dt.strftime('%Y-%m-%dT%H:%M:%S')
        weather_df.drop(columns=[orig_ts_col_name], inplace=True)
        print(f"    Processed 'time_iso' from '{orig_ts_col_name}'.")
        return weather_df

    def _weather_process_wind_direction(self, weather_df):
        print("  Processing wind direction...")
        def map_wind_norm(txt):
            if pd.isna(txt): return np.nan
            piece = re.sub(r'[^A-Za-z\\ -]','',str(txt).split('from the')[-1]).lower().strip()
            deg = {'north':0,'northeast':45,'east':90,'southeast':135,'south':180,'southwest':225,'west':270,'northwest':315}
            vals = [deg[t] for t in re.split('[\\\\-\\\\s]+',piece) if t in deg] 
            return np.mean(vals)/360 if vals else np.nan

        if 'Mean wind direction' in weather_df.columns:
            weather_df.loc[:, 'Mean wind direction'] = weather_df['Mean wind direction'].apply(map_wind_norm)
            print("    Processed 'Mean wind direction'.")
        elif 'DD' in weather_df.columns: 
            print("    Found 'DD' column, renaming and processing as Mean wind direction.")
            weather_df.rename(columns={'DD': 'Mean wind direction'}, inplace=True)
            weather_df.loc[:, 'Mean wind direction'] = weather_df['Mean wind direction'].apply(map_wind_norm)
        else:
            print("    Warning: No 'Mean wind direction' or 'DD' column found.")
        return weather_df

    def _weather_process_visibility(self, weather_df):
        print("  Processing visibility...")
        vis_col_name = next((c for c in weather_df.columns if isinstance(c, str) and c.lower()=='visibility'), None)
        if vis_col_name:
            weather_df.loc[:, vis_col_name] = (weather_df[vis_col_name].astype(str)
                             .str.extract(r'(\d+\.?\d*)')[0].astype(float))
            vmin, vmax = weather_df[vis_col_name].min(), weather_df[vis_col_name].max()
            if pd.notna(vmin) and pd.notna(vmax) and vmax > vmin: 
                weather_df.loc[:, vis_col_name] = (weather_df[vis_col_name] - vmin)/(vmax-vmin)
            elif pd.notna(vmin) and pd.notna(vmax) and vmax == vmin: 
                 weather_df.loc[:, vis_col_name] = 0.0 if vmin != 0 else 0.0 
            else: 
                weather_df.loc[:, vis_col_name] = np.nan 
            print(f"    Processed visibility column: '{vis_col_name}'.")
        else:
            print("    Warning: No 'visibility' column found.")
        return weather_df, vis_col_name

    def _weather_process_pressure(self, weather_df):
        # This function is now effectively a no-op since pressure columns are dropped by _weather_drop_unwanted_columns
        # However, we can keep it to ensure no accidental processing if the drop logic changes.
        # Or, it can be removed if _weather_drop_unwanted_columns is guaranteed to remove it.
        # For safety, let's assume it might still be called and just pass through.
        print("  Skipping atmospheric pressure processing as it's targeted for removal.")
        # Find the column name if it exists, but don't process it.
        pressure_col_name_final = None
        possible_pressure_cols = ['atm pressure to sea level', 'atmospheric pressure reduced to mean sea level', 'Po', 'Pressure at sea level']
        for col in weather_df.columns:
            if isinstance(col, str):
                for p_name in possible_pressure_cols:
                    if p_name.lower() in col.lower():
                        pressure_col_name_final = col # Identify if it exists
                        break
                if pressure_col_name_final:
                    break
        if pressure_col_name_final and pressure_col_name_final in weather_df.columns:
             print(f"    Column '{pressure_col_name_final}' (related to atm pressure) was found but will be dropped or was already dropped.")
             # Ensure it's not accidentally renamed if it was missed by drop
             if pressure_col_name_final == 'atm_pressure_sea_level':
                 pass # Already correctly named for potential later exclusion if needed
             elif 'atm_pressure_sea_level' not in weather_df.columns: # Avoid creating it if original is different
                 pass # Do not rename or create
             pressure_col_name_final = 'atm_pressure_sea_level' # Standardize name for exclusion logic

        else:
            pressure_col_name_final = 'atm_pressure_sea_level' # Default name for exclusion logic
            print("    No atmospheric pressure column found or it was already dropped.")
        return weather_df, pressure_col_name_final


    @staticmethod
    def _parse_cloud_cover_value(text_entry):
        if pd.isna(text_entry) or not isinstance(text_entry, str) or text_entry.strip() == "": return np.nan
        text_entry_lower = text_entry.lower().strip() 
        if "no significant cloud" in text_entry_lower or "sky clear" in text_entry_lower or text_entry_lower == "nsc": return 0.0
        if "vertical visibility" in text_entry_lower: return 100.0
        max_coverage = 0.0 
        layers = text_entry_lower.split(',') 
        found_specific_cloud_pattern = False
        for layer_text in layers:
            layer_text = layer_text.strip()
            current_layer_coverage = 0.0 
            layer_pattern_matched = False
            if "overcast (100%)" in layer_text or ("overcast" in layer_text and "100%" in layer_text) :
                current_layer_coverage = 100.0; layer_pattern_matched = True
            elif "broken clouds (60-90%)" in layer_text or ("broken" in layer_text and ("60-90%" in layer_text or "60%-90%" in layer_text)):
                current_layer_coverage = 75.0; layer_pattern_matched = True
            elif "scattered clouds (40-50%)" in layer_text or ("scattered" in layer_text and ("40-50%" in layer_text or "40%-50%" in layer_text)):
                current_layer_coverage = 45.0; layer_pattern_matched = True
            elif "few clouds (10-30%)" in layer_text or ("few" in layer_text and ("10-30%" in layer_text or "10%-30%" in layer_text)):
                current_layer_coverage = 20.0; layer_pattern_matched = True
            else: 
                percentages = re.findall(r'\((\d{1,3})(?:-(\d{1,3}))?%\)', layer_text)
                if percentages:
                    layer_pattern_matched = True 
                    for p_match in percentages:
                        if p_match[1]: low, high = int(p_match[0]), int(p_match[1]); current_layer_coverage = max(current_layer_coverage, (low + high) / 2.0)
                        elif p_match[0]: current_layer_coverage = max(current_layer_coverage, float(p_match[0]))
                elif 'oktas' in layer_text: 
                    okta_match = re.search(r'(\d)\s*oktas', layer_text)
                    if okta_match: oktas = int(okta_match.group(1)); current_layer_coverage = max(current_layer_coverage, oktas * 12.5); layer_pattern_matched = True
            if layer_pattern_matched: found_specific_cloud_pattern = True
            max_coverage = max(max_coverage, current_layer_coverage)
        if not found_specific_cloud_pattern and text_entry_lower and max_coverage == 0: return np.nan 
        return max_coverage

    def _weather_process_cloud_cover(self, weather_df):
        print("  Processing total cloud cover...")
        cloud_cover_col_name_orig = None
        possible_cloud_cols = ['total cloud cover', 'c', 'N', 'Cloud cover (total)'] 
        for col in weather_df.columns:
            if isinstance(col, str):
                for cc_name in possible_cloud_cols:
                    if cc_name.lower() == col.strip().lower(): cloud_cover_col_name_orig = col; break
                    elif cc_name.lower() in col.lower() and len(cc_name) > 2: cloud_cover_col_name_orig = col; break
                if cloud_cover_col_name_orig: break
        
        if cloud_cover_col_name_orig:
            print(f"    Processing total cloud cover column: '{cloud_cover_col_name_orig}'.")
            weather_df.loc[:, 'cloud_cover_percentage'] = weather_df[cloud_cover_col_name_orig].apply(WeatherCleaner._parse_cloud_cover_value)
        else:
            print("    Warning: Could not find 'total cloud cover' column. 'cloud_cover_percentage' will be NaN.")
            weather_df.loc[:, 'cloud_cover_percentage'] = np.nan 
        return weather_df, cloud_cover_col_name_orig

    def _weather_process_phenomena(self, weather_df):
        print("  Processing weather phenomena...")
        phenomena_col_ww = None; phenomena_col_w1w2 = None
        for col in weather_df.columns:
            if isinstance(col, str):
                if col.strip().upper() == 'WW': phenomena_col_ww = col
                elif "w'w'" in col.strip().lower() or "w1w2" in col.strip().lower(): phenomena_col_w1w2 = col
        
        weather_df.loc[:, 'phenomena_text'] = ""
        if phenomena_col_ww:
            print(f"    Processing weather phenomena column: '{phenomena_col_ww}' (WW).")
            weather_df.loc[:, 'phenomena_text'] = weather_df[phenomena_col_ww].astype(str).str.lower().str.strip().replace({'nan':'', 'none':''})
        else: print("    Warning: Could not find 'WW' (weather phenomena) column.")

        if phenomena_col_w1w2:
            print(f"    Processing weather phenomena column: '{phenomena_col_w1w2}' (W'W').")
            temp_phenomena_w1w2 = weather_df[phenomena_col_w1w2].astype(str).str.lower().str.strip().replace({'nan':'', 'none':''})
            weather_df.loc[:, 'phenomena_text'] = weather_df['phenomena_text'].astype(str).str.strip() + temp_phenomena_w1w2.apply(lambda x: (' ' + x.strip()) if x.strip() else '')
            weather_df.loc[:, 'phenomena_text'] = weather_df['phenomena_text'].str.strip()
        else: print("    Warning: Could not find 'W'W'' (secondary weather phenomena) column.")
        
        weather_df.loc[:, 'phenomena_text'] = weather_df['phenomena_text'].fillna('')
        return weather_df, phenomena_col_ww, phenomena_col_w1w2

    def _weather_convert_remaining_to_numeric(self, weather_df, cols_to_exclude):
        print("  Converting remaining weather columns to numeric...")
        numeric_cols_to_convert = weather_df.columns.difference(cols_to_exclude)
        for c in numeric_cols_to_convert:
            if c in weather_df.columns: 
                weather_df.loc[:, c] = pd.to_numeric(weather_df[c], errors='coerce')
        print(f"    Converted columns: {list(numeric_cols_to_convert)} (if they existed and weren't excluded).")
        return weather_df

    def _weather_interpolate_and_fill(self, weather_df_in):
        print("  Interpolating and filling NaNs in weather data...")
        weather_df = weather_df_in.copy()
        if 'time_iso' in weather_df.columns:
            weather_df['Datetime'] = pd.to_datetime(weather_df['time_iso'], errors='coerce')
            weather_df.dropna(subset=['Datetime'], inplace=True)
            if not weather_df.empty and not weather_df['Datetime'].isnull().all():
                weather_df_indexed = weather_df.set_index('Datetime')
                numeric_cols_for_interp = weather_df_indexed.select_dtypes(include=np.number).columns
                weather_df_indexed[numeric_cols_for_interp] = weather_df_indexed[numeric_cols_for_interp].interpolate(method='time')
                weather_df_interpolated = weather_df_indexed.reset_index()
                if 'time_iso' not in weather_df_interpolated.columns and 'Datetime' in weather_df_interpolated.columns:
                     weather_df_interpolated['time_iso'] = weather_df_interpolated['Datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')
                if 'time_iso' in weather_df.columns and 'time_iso' in weather_df_interpolated.columns:
                    weather_df_temp = weather_df.set_index('time_iso')
                    update_df_temp = weather_df_interpolated.set_index('time_iso')[[col for col in numeric_cols_for_interp if col in weather_df_interpolated.columns]]
                    weather_df_temp.update(update_df_temp) 
                    weather_df = weather_df_temp.reset_index()
                    print("    Time-weighted interpolation applied to numeric columns.")
                else:
                    print("    Warning: Could not reliably merge back interpolated data. Using ffill/bfill for numeric.")
                    numeric_cols_to_fill = weather_df.select_dtypes(include=np.number).columns
                    weather_df[numeric_cols_to_fill] = weather_df[numeric_cols_to_fill].ffill().bfill()
            else:
                print("    Warning: Could not set valid DatetimeIndex. Using ffill/bfill for numeric data.")
                numeric_cols_to_fill = weather_df.select_dtypes(include=np.number).columns
                weather_df[numeric_cols_to_fill] = weather_df[numeric_cols_to_fill].ffill().bfill()
        else:
            print("    Warning: 'time_iso' column not found. Using ffill/bfill for numeric data.")
            numeric_cols_to_fill = weather_df.select_dtypes(include=np.number).columns
            weather_df[numeric_cols_to_fill] = weather_df[numeric_cols_to_fill].ffill().bfill()

        string_cols_final = weather_df.select_dtypes(include='object').columns 
        weather_df[string_cols_final] = weather_df[string_cols_final].ffill().bfill()
        numeric_cols_final_fill = weather_df.select_dtypes(include=np.number).columns
        weather_df[numeric_cols_final_fill] = weather_df[numeric_cols_final_fill].ffill().bfill()
        return weather_df

    def process(self):
        """Main method to clean the weather data."""
        weather_df = self._read_data()
        if weather_df is None: return None

        weather_df = self._weather_drop_unwanted_columns(weather_df) # This now drops pressure and dewpoint
        weather_df = self._weather_process_timestamp(weather_df)
        if weather_df.empty: 
            print("Weather data became empty after timestamp processing. Halting.")
            return None 

        weather_df = self._weather_process_wind_direction(weather_df)
        weather_df, vis_col_name = self._weather_process_visibility(weather_df)
        
        # Pressure processing is effectively skipped/handled by the drop function
        # We still call it to get a consistent name for exclusion if it somehow wasn't dropped.
        weather_df, pressure_col_name_processed = self._weather_process_pressure(weather_df) 
        
        weather_df, cloud_cover_col_name_original = self._weather_process_cloud_cover(weather_df)
        weather_df, phenom_ww_original, phenom_w1w2_original = self._weather_process_phenomena(weather_df)

        cols_to_exclude = ['time_iso', 'phenomena_text'] 
        if 'Mean wind direction' in weather_df.columns: cols_to_exclude.append('Mean wind direction')
        if vis_col_name and vis_col_name in weather_df.columns: cols_to_exclude.append(vis_col_name)
        
        # Add the standard name for pressure to exclusion list, even if it was dropped.
        # This ensures it's not accidentally processed if drop logic changes or misses a variant.
        cols_to_exclude.append('atm_pressure_sea_level') 
        # Add dewpoint temperature variants to exclusion if they were not caught by drop
        # (though the drop function should be more robust now)
        for col in weather_df.columns:
            if isinstance(col, str) and "dewpoint temperature" in col.lower():
                if col not in cols_to_exclude: cols_to_exclude.append(col)

        if 'cloud_cover_percentage' in weather_df.columns: cols_to_exclude.append('cloud_cover_percentage') 
        if cloud_cover_col_name_original and cloud_cover_col_name_original in weather_df.columns and cloud_cover_col_name_original != 'cloud_cover_percentage' : 
            cols_to_exclude.append(cloud_cover_col_name_original) 
        if phenom_ww_original and phenom_ww_original in weather_df.columns: cols_to_exclude.append(phenom_ww_original)
        if phenom_w1w2_original and phenom_w1w2_original in weather_df.columns: cols_to_exclude.append(phenom_w1w2_original)
        
        weather_df = self._weather_convert_remaining_to_numeric(weather_df, list(set(cols_to_exclude))) 

        weather_df = weather_df.sort_values('time_iso').reset_index(drop=True)
        weather_df = self._weather_interpolate_and_fill(weather_df) 

        desired_order = ['time_iso', 'phenomena_text', 'cloud_cover_percentage']
        # No need to add pressure_col_name_processed to desired_order as it's being dropped
        
        remaining_cols = [c for c in weather_df.columns if c not in desired_order and c != 'Datetime'] 
        final_cols_order = desired_order + sorted([c for c in remaining_cols if c in weather_df.columns]) 
        final_cols_order = [c for c in final_cols_order if c in weather_df.columns] 
        weather_df = weather_df[final_cols_order]
        
        weather_df = WeatherCleaner._bump_zeros(weather_df) 
        print(f"Weather data cleaning for sheet '{self.sheet_name}' complete.")
        return weather_df

