# eda.py
# Energy Consumption Forecasting
# Section 3: Exploratory Data Analysis (Enhanced Profiling)

import pandas as pd
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import math 
import os 

# Define the names of the input CSV files (output from data_cleaning.py)
INPUT_DATA_DIR = 'data'
ELECTRICITY_CSV = os.path.join(INPUT_DATA_DIR, 'cleaned_electricity.csv')
AREAS_CSV = os.path.join(INPUT_DATA_DIR, 'cleaned_areas.csv')
WEATHER_CSV = os.path.join(INPUT_DATA_DIR, 'cleaned_weather.csv') # For temperature correlation

# Define output directories for plots and potentially data
PROFILING_DIR = 'profiling'
PLOT_OUTPUT_DIR = os.path.join(PROFILING_DIR, 'plots')
BASE_DATA_OUTPUT_DIR = os.path.join(PROFILING_DIR, 'data') 
FIGURE_DPI = 150 

def ensure_output_dir(directory_name):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"Created directory: {directory_name}")

def load_cleaned_data():
    """Loads cleaned electricity, areas, and weather data from CSV files."""
    print("Loading cleaned data for EDA...")
    elec_df, areas_df, weather_df, area_map = None, None, None, {}
    
    try:
        elec_df = pd.read_csv(ELECTRICITY_CSV)
        elec_df['Datetime'] = pd.to_datetime(elec_df['time_iso'])
        elec_df = elec_df.set_index('Datetime', drop=False) 
        print(f"  Electricity data loaded. Index range: {elec_df.index.min()} to {elec_df.index.max()}")
        print(f"  Elec NaT count in index: {elec_df.index.isna().sum()}")
    except FileNotFoundError:
        print(f"Error: Electricity CSV file not found at '{ELECTRICITY_CSV}'.")
        return None, None, None, {}
    except Exception as e:
        print(f"Error loading electricity data: {e}")
        return None, None, None, {}


    try:
        areas_df = pd.read_csv(AREAS_CSV)
        if 'Buid_ID' in areas_df.columns and 'Area [m2]' in areas_df.columns:
            area_map = areas_df.set_index('Buid_ID')['Area [m2]'].to_dict()
        else:
            print("Warning: 'Buid_ID' or 'Area [m2]' not found in areas CSV. Area map will be empty.")
    except FileNotFoundError:
        print(f"Warning: Areas CSV file not found at '{AREAS_CSV}'. Area information will be missing.")
    except Exception as e:
        print(f"Error loading areas data: {e}")


    try:
        weather_df = pd.read_csv(WEATHER_CSV)
        weather_df['Datetime'] = pd.to_datetime(weather_df['time_iso'])
        weather_df = weather_df.set_index('Datetime', drop=False) 
        print(f"  Weather data loaded. Index range: {weather_df.index.min()} to {weather_df.index.max()}")
        print(f"  Weather NaT count in index: {weather_df.index.isna().sum()}")
        # print("Weather data loaded.") # Redundant print
    except FileNotFoundError:
        print(f"Warning: Weather CSV file not found at '{WEATHER_CSV}'. Weather-related plots will be skipped.")
        weather_df = None 
    except Exception as e:
        print(f"Error loading weather data: {e}")
        weather_df = None
        
    print("Cleaned data loading step complete.")
    return elec_df, areas_df, weather_df, area_map

def plot_electricity_consumption_over_time(elec_df, area_map_dict):
    if elec_df is None:
        print("Electricity data not loaded. Skipping 'consumption over time' plot.")
        return
        
    print("Plotting electricity consumption over time...")
    plot_dir = os.path.join(PLOT_OUTPUT_DIR, "overall_consumption")
    ensure_output_dir(plot_dir) 
    
    plt.figure(figsize=(20, 10), dpi=FIGURE_DPI)
    plot_columns = [col for col in elec_df.columns if col not in ['time_iso', 'Datetime']] 
    for col in plot_columns:
        if col in elec_df:
            area = area_map_dict.get(col, None)
            lbl = f"{col} ({int(area) if pd.notnull(area) else 'N/A'} m²)" if area is not None else col
            plt.plot(elec_df.index, elec_df[col], label=lbl, lw=1, alpha=0.8) 

    plt.title("Electricity Consumption Over the Year by Building", fontsize=18)
    plt.xlabel("Date", fontsize=14); plt.ylabel("kWh", fontsize=14)
    plt.legend(fontsize='medium', ncol=max(1, len(plot_columns) // 8))
    plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(pad=1.5)
    
    filename = os.path.join(plot_dir, "overall_consumption_by_building.png")
    plt.savefig(filename); print(f"Plot saved: {filename}"); plt.close()

def plot_hourly_profile_per_building(elec_df, area_map_dict):
    if elec_df is None: return
    profile_type_name = 'hourly_per_building'
    print(f"Plotting and saving data for average {profile_type_name} consumption profile...")
    
    data_out_dir = os.path.join(BASE_DATA_OUTPUT_DIR, profile_type_name)
    plot_out_dir = os.path.join(PLOT_OUTPUT_DIR, profile_type_name)
    ensure_output_dir(data_out_dir); ensure_output_dir(plot_out_dir)

    df_for_grouping = elec_df.drop(columns=['time_iso', 'Datetime'], errors='ignore') 
    building_cols = df_for_grouping.select_dtypes(include=np.number).columns.tolist()
    if not building_cols: print(f"No building/numeric columns for {profile_type_name} profile."); return

    num_buildings = len(building_cols)
    plt.figure(figsize=(18, 9), dpi=FIGURE_DPI) 
    palette = sns.color_palette("husl", num_buildings)
    all_profiles = {}

    for i, building in enumerate(building_cols):
        if building in df_for_grouping: 
            profile = df_for_grouping[building].groupby(df_for_grouping.index.hour).mean()
            all_profiles[building] = profile
            area = area_map_dict.get(building, None)
            lbl = f"{building} ({int(area)} m²)" if pd.notnull(area) else building
            plt.plot(profile.index, profile, lw=2, label=lbl, color=palette[i % len(palette)])
            profile.to_csv(os.path.join(data_out_dir, f"hourly_profile_{building}.csv"), header=[f'{building}_avg_kWh'])
            
    plt.title("Average Consumption by Hour of Day (Per Building)", fontsize=20)
    plt.xlabel("Hour of Day", fontsize=16); plt.ylabel("Average kWh", fontsize=16)
    plt.xticks(range(0, 24)); plt.legend(title="Building", fontsize='medium', ncol=max(1, num_buildings // 5))
    plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(pad=1.5)
    
    plot_filename = os.path.join(plot_out_dir, "hourly_profile_all_buildings_combined.png")
    plt.savefig(plot_filename); print(f"Plot saved: {plot_filename}"); plt.close()
    print(f"{profile_type_name.capitalize()} data tables saved to {data_out_dir}")


def plot_generic_profiles_per_building(elec_df, area_map_dict, profile_type, group_by_attr, x_label, legend_map=None, palette_name="viridis", num_legend_cols=2):
    if elec_df is None: print(f"Electricity data not loaded. Skipping '{profile_type}' plot."); return

    print(f"Plotting and saving data for average {profile_type} consumption profile (per building)...")
    data_out_dir = os.path.join(BASE_DATA_OUTPUT_DIR, profile_type + "_per_building")
    plot_out_dir = os.path.join(PLOT_OUTPUT_DIR, profile_type + "_per_building")
    ensure_output_dir(data_out_dir); ensure_output_dir(plot_out_dir)
    
    building_cols = [col for col in elec_df.columns if col not in ['time_iso', 'Datetime'] and pd.api.types.is_numeric_dtype(elec_df[col])]
    if not building_cols: print(f"No building columns for {profile_type} profile."); return

    num_buildings = len(building_cols)
    cols_subplot = math.ceil(math.sqrt(num_buildings)); rows_subplot = math.ceil(num_buildings / cols_subplot)
    fig_width = max(18, 6 * cols_subplot); fig_height = max(15, 5.5 * rows_subplot)
    fig, axes = plt.subplots(rows_subplot, cols_subplot, figsize=(fig_width, fig_height), squeeze=False, dpi=FIGURE_DPI)
    axes = axes.flatten() 

    for i, building_col in enumerate(building_cols):
        ax = axes[i]
        building_data_for_profile = elec_df[[building_col]].copy() 
        
        if profile_type == 'seasonal':
            def get_season(date):
                month = date.month
                if month in [12, 1, 2]: return 'Winter'
                elif month in [3, 4, 5]: return 'Spring'
                elif month in [6, 7, 8]: return 'Summer'
                else: return 'Autumn'
            building_data_for_profile['group_attr'] = building_data_for_profile.index.to_series().apply(get_season)
        else:
            building_data_for_profile['group_attr'] = getattr(building_data_for_profile.index, group_by_attr)

        profile_table = building_data_for_profile.groupby(['group_attr', building_data_for_profile.index.hour])[building_col].mean().unstack(level=0)
        
        if legend_map: profile_table.columns = [legend_map.get(c, c) for c in profile_table.columns]
        
        if profile_type == 'seasonal': 
            season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
            ordered_columns = [s for s in season_order if s in profile_table.columns]
            profile_table = profile_table.reindex(columns=ordered_columns) 

        if not profile_table.empty:
            profile_table.to_csv(os.path.join(data_out_dir, f"{profile_type}_profile_{building_col}.csv"))
            palette = sns.color_palette(palette_name, n_colors=len(profile_table.columns))
            for j, col_name in enumerate(profile_table.columns):
                if col_name in profile_table: 
                    ax.plot(profile_table.index, profile_table[col_name], label=col_name, lw=1.5, color=palette[j % len(palette)])
        else: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=10)
        
        area = area_map_dict.get(building_col, "N/A"); ax.set_title(f"{building_col} (Area: {area} m²)", fontsize=12) 
        ax.set_xlabel("Hour of Day", fontsize=10); ax.set_ylabel("Avg kWh", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9); ax.legend(title=x_label, fontsize='small', ncol=num_legend_cols) 
        ax.grid(True, linestyle='--', alpha=0.6)

    for j in range(num_buildings, len(axes)): fig.delaxes(axes[j])
    fig.suptitle(f"Average Hourly Consumption by {x_label} (Per Building)", fontsize=20, y=1.0) 
    fig.tight_layout(rect=[0, 0.02, 1, 0.97]) 
    plot_filename = os.path.join(plot_out_dir, f"{profile_type}_consumption_profile_per_building.png")
    plt.savefig(plot_filename); print(f"Plot saved: {plot_filename}"); plt.close(fig) 
    print(f"{profile_type.capitalize()} per-building profile data tables saved to {data_out_dir}")

def plot_consumption_distribution_by_dow(elec_df, area_map_dict):
    if elec_df is None: print("Electricity data not loaded. Skipping DOW distribution plot."); return
    print("Plotting consumption distribution by day of week (per building)...")
    
    plot_out_dir = os.path.join(PLOT_OUTPUT_DIR, "distribution_by_dow")
    ensure_output_dir(plot_out_dir)
    
    building_cols = [col for col in elec_df.columns if col not in ['time_iso', 'Datetime'] and pd.api.types.is_numeric_dtype(elec_df[col])]
    if not building_cols: print("No building columns for DOW distribution plot."); return

    num_buildings = len(building_cols)
    cols_subplot = math.ceil(math.sqrt(num_buildings)); rows_subplot = math.ceil(num_buildings / cols_subplot)
    fig_width = max(18, 5 * cols_subplot); fig_height = max(15, 5 * rows_subplot)
    fig, axes = plt.subplots(rows_subplot, cols_subplot, figsize=(fig_width, fig_height), squeeze=False, dpi=FIGURE_DPI)
    axes = axes.flatten()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    for i, building_col in enumerate(building_cols):
        ax = axes[i]
        building_data_for_plot = elec_df[[building_col]].copy() 
        building_data_for_plot['day_of_week_num'] = building_data_for_plot.index.dayofweek
        
        sns.boxplot(x='day_of_week_num', y=building_col, data=building_data_for_plot, ax=ax, 
                    hue='day_of_week_num', palette="viridis", showfliers=False, legend=False) 
        
        ax.set_xticks(range(len(day_names))) 
        ax.set_xticklabels(day_names) 
        area = area_map_dict.get(building_col, "N/A"); ax.set_title(f"{building_col} (Area: {area} m²)", fontsize=12)
        ax.set_xlabel("Day of Week", fontsize=10); ax.set_ylabel("Hourly kWh", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9); ax.grid(True, linestyle=':', alpha=0.5)

    for j in range(num_buildings, len(axes)): fig.delaxes(axes[j])
    fig.suptitle("Hourly Consumption Distribution by Day of Week (Per Building)", fontsize=20, y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    plot_filename = os.path.join(plot_out_dir, "consumption_distribution_by_dow.png")
    plt.savefig(plot_filename); print(f"Plot saved: {plot_filename}"); plt.close(fig)

def plot_hourly_profile_per_sqm(elec_df, area_map_dict):
    if elec_df is None or not area_map_dict: print("Elec data or area map missing. Skipping consumption/m² plot."); return
    print("Plotting hourly consumption per square meter (per building)...")

    data_out_dir = os.path.join(BASE_DATA_OUTPUT_DIR, "hourly_per_sqm")
    plot_out_dir = os.path.join(PLOT_OUTPUT_DIR, "hourly_per_sqm")
    ensure_output_dir(data_out_dir); ensure_output_dir(plot_out_dir)

    building_cols = [col for col in elec_df.columns if col not in ['time_iso', 'Datetime'] and pd.api.types.is_numeric_dtype(elec_df[col])]
    if not building_cols: print("No building columns for consumption/m² plot."); return
    
    plt.figure(figsize=(18, 9), dpi=FIGURE_DPI)
    palette = sns.color_palette("coolwarm", len(building_cols))
    
    for i, building_col in enumerate(building_cols):
        area = area_map_dict.get(building_col)
        if area and area > 0:
            consumption_per_sqm = elec_df[building_col] / area
            hourly_profile_per_sqm = consumption_per_sqm.groupby(consumption_per_sqm.index.hour).mean()
            hourly_profile_per_sqm.to_csv(os.path.join(data_out_dir, f"hourly_profile_per_sqm_{building_col}.csv"), header=[f'{building_col}_avg_kWh_per_m2'])
            plt.plot(hourly_profile_per_sqm.index, hourly_profile_per_sqm, label=f"{building_col}", lw=2, color=palette[i % len(palette)])
        else:
            print(f"  Skipping {building_col} for consumption/m² plot due to missing or zero area.")
            
    plt.title("Average Hourly Consumption per m² (Per Building)", fontsize=20)
    plt.xlabel("Hour of Day", fontsize=16); plt.ylabel("Average kWh / m²", fontsize=16)
    plt.xticks(range(0, 24)); plt.legend(title="Building", fontsize='medium', ncol=max(1, len(building_cols) // 5))
    plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(pad=1.5)
    plot_filename = os.path.join(plot_out_dir, "hourly_profile_per_sqm_all_buildings.png")
    plt.savefig(plot_filename); print(f"Plot saved: {plot_filename}"); plt.close()
    print(f"Hourly consumption/m² data tables saved to {data_out_dir}")

def plot_consumption_vs_temperature(elec_df, weather_df_original, area_map_dict): # Renamed weather_df to weather_df_original
    if elec_df is None or weather_df_original is None: print("Elec or weather data missing. Skipping consumption vs temp plot."); return
    
    print("Plotting consumption vs. temperature (per building)...")
    plot_out_dir = os.path.join(PLOT_OUTPUT_DIR, "consumption_vs_temperature")
    ensure_output_dir(plot_out_dir)

    temp_col_name = None
    possible_temp_names = ['air temperature', 'temperature', 'temp', 't'] 
    weather_df_cols_lower = {col.strip().lower(): col for col in weather_df_original.columns if isinstance(col, str)}

    for p_name in possible_temp_names:
        if p_name in weather_df_cols_lower:
            temp_col_name = weather_df_cols_lower[p_name] 
            break
            
    if not temp_col_name: print("  Temperature column not found in weather data. Skipping plot."); return
    print(f"  Using '{temp_col_name}' as temperature column.")

    # --- RESAMPLE WEATHER DATA TO HOURLY ---
    # Ensure weather_df_original has DatetimeIndex if not already set (load_cleaned_data should do this)
    weather_df_resampled = weather_df_original.copy()
    if not isinstance(weather_df_resampled.index, pd.DatetimeIndex):
        if 'Datetime' in weather_df_resampled.columns:
            weather_df_resampled = weather_df_resampled.set_index('Datetime', drop=False)
        else:
            print("  Weather data does not have a 'Datetime' column or index for resampling. Skipping plot.")
            return
            
    # Resample the specific temperature column and any other needed numeric columns
    # For simplicity, just resampling the temp_col_name. If other weather features are needed, resample them too.
    weather_to_join = weather_df_resampled[[temp_col_name]].resample('h').mean() 
    weather_to_join.ffill(inplace=True) # Fill any NaNs created by resampling (e.g., if an hour had no data)
    weather_to_join.bfill(inplace=True)
    print(f"  Weather data resampled to hourly. Resampled index sample: {weather_to_join.index[:5]}")
    # --- END RESAMPLE ---

    print(f"  Elec index sample before merge: {elec_df.index[:5]}")
    print(f"  Resampled Weather index sample before merge: {weather_to_join.index[:5]}")

    merged_df = elec_df.join(weather_to_join, how='inner') # Join with resampled weather
    
    if merged_df.empty: 
        print("  No common timestamps between electricity and RESAMPLED weather data after merge. Skipping plot.")
        print(f"  Elec index min/max: {elec_df.index.min()} / {elec_df.index.max()}")
        print(f"  Resampled Weather index min/max: {weather_to_join.index.min()} / {weather_to_join.index.max()}")
        print(f"  Elec index timezone: {elec_df.index.tz}")
        print(f"  Resampled Weather index timezone: {weather_to_join.index.tz}")
        return

    building_cols = [col for col in elec_df.columns if col not in ['time_iso', 'Datetime'] and pd.api.types.is_numeric_dtype(elec_df[col])]
    if not building_cols: print("No building columns for consumption vs temp plot."); return

    num_buildings = len(building_cols)
    cols_subplot = math.ceil(math.sqrt(num_buildings)); rows_subplot = math.ceil(num_buildings / cols_subplot)
    fig_width = max(18, 6 * cols_subplot); fig_height = max(15, 5.5 * rows_subplot)
    fig, axes = plt.subplots(rows_subplot, cols_subplot, figsize=(fig_width, fig_height), squeeze=False, dpi=FIGURE_DPI)
    axes = axes.flatten()

    for i, building_col in enumerate(building_cols):
        ax = axes[i]
        if building_col in merged_df and temp_col_name in merged_df:
            sample_df = merged_df[[building_col, temp_col_name]].dropna()
            if len(sample_df) > 5000: 
                sample_df = sample_df.sample(n=5000, random_state=1)

            sns.scatterplot(x=temp_col_name, y=building_col, data=sample_df, ax=ax, alpha=0.5, s=10) 
            area = area_map_dict.get(building_col, "N/A"); ax.set_title(f"{building_col} (Area: {area} m²)", fontsize=12)
            ax.set_xlabel(f"Temperature ({temp_col_name})", fontsize=10); ax.set_ylabel("Hourly kWh", fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=9); ax.grid(True, linestyle=':', alpha=0.5)
        else: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=10)

    for j in range(num_buildings, len(axes)): fig.delaxes(axes[j])
    fig.suptitle("Hourly Consumption vs. Temperature (Per Building)", fontsize=20, y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    plot_filename = os.path.join(plot_out_dir, "consumption_vs_temperature.png")
    plt.savefig(plot_filename); print(f"Plot saved: {plot_filename}"); plt.close(fig)


def main():
    """Main function to run EDA."""
    ensure_output_dir(PLOT_OUTPUT_DIR) 
    ensure_output_dir(BASE_DATA_OUTPUT_DIR) 

    elec, areas, weather, area_map = load_cleaned_data() 

    if elec is not None:
        elec_for_plots = elec.drop(columns=['time_iso', 'Datetime'], errors='ignore') 
        
        plot_electricity_consumption_over_time(elec_for_plots.copy(), area_map)
        plot_hourly_profile_per_building(elec_for_plots.copy(), area_map) 

        day_names_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        month_names_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                           7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        
        plot_generic_profiles_per_building(elec_for_plots.copy(), area_map, 'weekly', 'dayofweek', 'Day of Week', legend_map=day_names_map, palette_name="viridis", num_legend_cols=3)
        plot_generic_profiles_per_building(elec_for_plots.copy(), area_map, 'monthly', 'month', 'Month', legend_map=month_names_map, palette_name="Spectral", num_legend_cols=4)
        plot_generic_profiles_per_building(elec_for_plots.copy(), area_map, 'seasonal', 'season', 'Season', palette_name="coolwarm", num_legend_cols=2) 

        plot_consumption_distribution_by_dow(elec_for_plots.copy(), area_map)
        plot_hourly_profile_per_sqm(elec_for_plots.copy(), area_map)
        if weather is not None: 
            plot_consumption_vs_temperature(elec_for_plots.copy(), weather.copy(), area_map)
        else:
            print("Skipping consumption vs. temperature plots as weather data is not available.")
    else:
        print("EDA cannot proceed as electricity data failed to load.")

if __name__ == '__main__':
    main()
