# eda.py
# Energy Consumption Forecasting
# Section 3: Exploratory Data Analysis (Per-Building Profiles with Save & Organized Data Output)

# This script will explore:
# - Aggregate load patterns: hourly, daily, weekly, monthly, seasonal cycles FOR EACH BUILDING.
# - Building-level comparisons: which buildings have highest peaks.
# - Plots will be saved to 'profiling/plots/' directory.
# - Profiled data (e.g., mean consumption tables) will be saved to 'profiling/data/[profile_type]/'.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # For potentially nicer plots and color palettes
import math # For calculating subplot grid
import os # For creating output directory

# Define the names of the input CSV files (output from data_cleaning.py)
ELECTRICITY_CSV = os.path.join('data', 'cleaned_electricity.csv') # Corrected path assuming 'data' subfolder for inputs
AREAS_CSV = os.path.join('data', 'cleaned_areas.csv') # Corrected path

# Define output directories for plots and potentially data
PROFILING_DIR = 'profiling'
PLOT_OUTPUT_DIR = os.path.join(PROFILING_DIR, 'plots')
BASE_DATA_OUTPUT_DIR = os.path.join(PROFILING_DIR, 'data') # Base directory for profiled data tables
FIGURE_DPI = 300 # Set DPI for saved figures

def ensure_output_dir(directory_name):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"Created directory: {directory_name}")

def load_cleaned_data():
    """Loads cleaned data from CSV files."""
    print("Loading cleaned data for EDA...")
    try:
        elec = pd.read_csv(ELECTRICITY_CSV)
        areas = pd.read_csv(AREAS_CSV)
    except FileNotFoundError as e:
        print(f"Error: One or more cleaned CSV files not found. Please run data_cleaning.py first.")
        print(f"Expected at: {ELECTRICITY_CSV}, {AREAS_CSV}")
        print(f"Details: {e}")
        return None, None, None 

    elec['Datetime'] = pd.to_datetime(elec['time_iso'])
    elec.set_index('Datetime', inplace=True, drop=False) 
    
    if 'Buid_ID' in areas.columns and 'Area [m2]' in areas.columns:
        area_map = areas.set_index('Buid_ID')['Area [m2]'].to_dict()
    else:
        print("Warning: 'Buid_ID' or 'Area [m2]' not found in areas CSV. Area map will be empty.")
        area_map = {}
        
    print("Cleaned data loaded.")
    return elec, areas, area_map

def plot_electricity_consumption_over_time(elec_df, area_map_dict):
    """Plots electricity consumption over time by building and saves it."""
    if elec_df is None:
        print("Electricity data not loaded. Skipping 'consumption over time' plot.")
        return
        
    print("Plotting electricity consumption over time...")
    ensure_output_dir(PLOT_OUTPUT_DIR) # Ensure plot directory exists
    plt.figure(figsize=(20, 10), dpi=FIGURE_DPI)
    
    plot_columns = [col for col in elec_df.columns if col not in ['time_iso', 'Datetime']]

    for col in plot_columns:
        if col in elec_df:
            area = area_map_dict.get(col, None)
            lbl = f"{col} ({int(area) if pd.notnull(area) else 'N/A'} m²)" if area is not None else col
            plt.plot(elec_df.index, elec_df[col], label=lbl, lw=1, alpha=0.8)

    plt.title("Electricity Consumption Over the Year by Building", fontsize=18)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("kWh", fontsize=14)
    plt.legend(fontsize='medium', ncol=max(1, len(plot_columns) // 8))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename = os.path.join(PLOT_OUTPUT_DIR, "electricity_consumption_over_time.png")
    plt.savefig(filename)
    print(f"Plot saved: {filename}")
    plt.close()

def plot_hourly_profile(elec_df, area_map_dict):
    """Plots average consumption by hour of day for each building and saves plot and data."""
    if elec_df is None:
        print("Electricity data not loaded. Skipping 'hourly profile' plot.")
        return

    profile_type_name = 'hourly'
    print(f"Plotting and saving data for average {profile_type_name} consumption profile (per building)...")
    
    # Define specific output directory for this profile type's data
    hourly_data_output_dir = os.path.join(BASE_DATA_OUTPUT_DIR, profile_type_name)
    ensure_output_dir(hourly_data_output_dir)
    ensure_output_dir(PLOT_OUTPUT_DIR) # Ensure plot directory exists

    df_for_grouping = elec_df.drop(columns=['time_iso', 'Datetime'], errors='ignore')
    building_cols = df_for_grouping.select_dtypes(include=np.number).columns.tolist()
    
    if not building_cols:
        print(f"No building/numeric columns found for {profile_type_name} profile.")
        return

    num_buildings = len(building_cols)
    if num_buildings == 0:
        print(f"No building columns to plot for {profile_type_name} profile.")
        return

    all_hourly_profiles_data = {} 

    plt.figure(figsize=(15, 8), dpi=FIGURE_DPI) 
    palette = sns.color_palette("husl", num_buildings)

    for i, building in enumerate(building_cols):
        if building in df_for_grouping:
            hourly_profile_building = df_for_grouping[building].groupby(df_for_grouping.index.hour).mean()
            all_hourly_profiles_data[building] = hourly_profile_building
            area = area_map_dict.get(building, None)
            lbl = f"{building} ({int(area)} m²)" if pd.notnull(area) else building
            plt.plot(hourly_profile_building.index, hourly_profile_building, lw=2, label=lbl, color=palette[i % len(palette)])
        
    plt.title("Average Consumption by Hour of Day (Per Building)", fontsize=18)
    plt.xlabel("Hour of Day", fontsize=14)
    plt.ylabel("Average kWh", fontsize=14)
    plt.xticks(range(0, 24))
    plt.legend(title="Building", fontsize='medium', ncol=max(1, num_buildings // 6)) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_filename = os.path.join(PLOT_OUTPUT_DIR, f"{profile_type_name}_consumption_profile_per_building.png")
    plt.savefig(plot_filename)
    print(f"Plot saved: {plot_filename}")
    plt.close()
    
    for bldg, profile_data in all_hourly_profiles_data.items():
        if not profile_data.empty:
            data_filename = os.path.join(hourly_data_output_dir, f"{profile_type_name}_profile_{bldg}.csv")
            profile_data.to_csv(data_filename, header=[f'{bldg}_avg_kWh'])
            print(f"Data saved: {data_filename}")
    print(f"{profile_type_name.capitalize()} profile data tables saved to {hourly_data_output_dir}")


def plot_per_building_profiles(elec_df, area_map_dict, profile_type):
    """
    Generic function to plot per-building profiles (weekly, monthly, seasonal) and save the figure and data.
    profile_type can be 'weekly', 'monthly', or 'seasonal'.
    """
    if elec_df is None:
        print(f"Electricity data not loaded. Skipping '{profile_type} profile' plot.")
        return

    print(f"Plotting and saving data for average {profile_type} consumption profile (per building)...")
    
    # Define specific output directory for this profile type's data and plots
    profile_specific_data_dir = os.path.join(BASE_DATA_OUTPUT_DIR, profile_type)
    ensure_output_dir(profile_specific_data_dir)
    ensure_output_dir(PLOT_OUTPUT_DIR) # Ensure plot directory exists
    
    building_cols = [col for col in elec_df.columns if col not in ['time_iso', 'Datetime'] and pd.api.types.is_numeric_dtype(elec_df[col])]

    if not building_cols:
        print(f"No building columns found to plot for {profile_type} profile.")
        return

    num_buildings = len(building_cols)
    
    cols_subplot = math.ceil(math.sqrt(num_buildings))
    rows_subplot = math.ceil(num_buildings / cols_subplot)

    fig_width = max(15, 5 * cols_subplot) 
    fig_height = max(12, 4 * rows_subplot)

    fig, axes = plt.subplots(rows_subplot, cols_subplot, figsize=(fig_width, fig_height), squeeze=False, dpi=FIGURE_DPI)
    axes = axes.flatten() 

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    month_names_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    
    all_profile_data_tables_for_type = {}

    def get_season(date):
        month = date.month
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Autumn'

    for i, building_col in enumerate(building_cols):
        ax = axes[i]
        building_data = elec_df[[building_col]].copy() 
        profile_table = pd.DataFrame()

        if profile_type == 'weekly':
            grouped = building_data.groupby([building_data.index.dayofweek, building_data.index.hour]).mean()
            if not grouped.empty and building_col in grouped:
                profile_table = grouped[building_col].unstack(level=0)
                if not profile_table.empty:
                     profile_table.columns = [day_names[d] for d in profile_table.columns]
            palette = sns.color_palette("viridis", 7)
            legend_title = "Day of Week"
        elif profile_type == 'monthly':
            grouped = building_data.groupby([building_data.index.month, building_data.index.hour]).mean()
            if not grouped.empty and building_col in grouped:
                profile_table = grouped[building_col].unstack(level=0)
                if not profile_table.empty:
                    # Ensure columns are sorted before mapping to names if they aren't already
                    sorted_month_indices = sorted(profile_table.columns)
                    profile_table = profile_table[sorted_month_indices] # Reorder columns by month index
                    profile_table.columns = [month_names_map[m] for m in sorted_month_indices]
            palette = sns.color_palette("Spectral", 12)
            legend_title = "Month"
        elif profile_type == 'seasonal':
            building_data['Season'] = building_data.index.to_series().apply(get_season)
            grouped = building_data.groupby(['Season', building_data.index.hour]).mean()
            if not grouped.empty and building_col in grouped:
                profile_table = grouped[building_col].unstack(level=0)
                if not profile_table.empty:
                    # Filter season_order to only include seasons present in profile_table.columns
                    present_seasons = [s for s in season_order if s in profile_table.columns]
                    profile_table = profile_table.reindex(columns=present_seasons) 
            palette = sns.color_palette("coolwarm", 4)
            legend_title = "Season"
        else:
            continue 
        
        all_profile_data_tables_for_type[building_col] = profile_table

        if profile_table.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=10)
        else:
            for j, col_name in enumerate(profile_table.columns):
                if col_name in profile_table: 
                    ax.plot(profile_table.index, profile_table[col_name], label=col_name, lw=1.5, color=palette[j % len(palette)])
        
        area = area_map_dict.get(building_col, "N/A")
        ax.set_title(f"{building_col} (Area: {area} m²)", fontsize=12) 
        ax.set_xlabel("Hour of Day", fontsize=10)
        ax.set_ylabel("Avg kWh", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.legend(title=legend_title, fontsize='small', ncol=2) 
        ax.grid(True, linestyle='--', alpha=0.6)

    for j in range(num_buildings, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Average Hourly Consumption by {profile_type.capitalize()} (Per Building)", fontsize=20, y=1.0) 
    fig.tight_layout(rect=[0, 0.02, 1, 0.97]) 
    
    plot_filename = os.path.join(PLOT_OUTPUT_DIR, f"{profile_type}_consumption_profile_per_building.png")
    plt.savefig(plot_filename)
    print(f"Plot saved: {plot_filename}")
    plt.close(fig) 

    for bldg, data_table in all_profile_data_tables_for_type.items():
        if not data_table.empty:
            data_filename = os.path.join(profile_specific_data_dir, f"{profile_type}_profile_{bldg}.csv")
            data_table.to_csv(data_filename)
            print(f"Data saved: {data_filename}")
    print(f"{profile_type.capitalize()} profile data tables saved to {profile_specific_data_dir}")


def main():
    """Main function to run EDA."""
    # Ensure base directories are created. Specific profile type data dirs are created within plotting functions.
    ensure_output_dir(PLOT_OUTPUT_DIR) 
    ensure_output_dir(BASE_DATA_OUTPUT_DIR) 

    elec, areas, area_map = load_cleaned_data()

    if elec is not None:
        elec_for_plots = elec.drop(columns=['time_iso'], errors='ignore')
        
        plot_electricity_consumption_over_time(elec_for_plots.copy(), area_map)
        plot_hourly_profile(elec_for_plots.copy(), area_map) 

        plot_per_building_profiles(elec_for_plots.copy(), area_map, 'weekly')
        plot_per_building_profiles(elec_for_plots.copy(), area_map, 'monthly')
        plot_per_building_profiles(elec_for_plots.copy(), area_map, 'seasonal')
    else:
        print("EDA cannot proceed as electricity data failed to load.")

if __name__ == '__main__':
    main()
