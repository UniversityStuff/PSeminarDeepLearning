import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
import glob
import geopandas as gpd


def load_landslide_data(shapefile_path):
    """
    Load landslide data from shapefile.
    
    Parameters:
    - shapefile_path: Path to the shapefile
    
    Returns:
    - DataFrame with landslide data
    """
    # Read shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Extract coordinates from geometry
    gdf['Longitude'] = gdf.geometry.x
    gdf['Latitude'] = gdf.geometry.y
    
    # Convert to regular DataFrame
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    
    # Convert date column to datetime
    date_column = None
    for col in ['Datum']:
        if col in df.columns:
            date_column = col
            break
    
    if date_column is None:
        raise ValueError(f"Could not find date column in shapefile. Available columns: {df.columns.tolist()}")
    
    df['Datum'] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Select relevant columns
    base_columns = ['Datum', 'Latitude', 'Longitude', 'UUID', 'MOVEMENT_C', 'MOVEMENT_D']
    
    columns = base_columns.copy()
    
    result_df = df[columns].copy()
    
    print(f"Loaded {len(result_df)} landslide events")
    return result_df


def find_nc_file_for_date(data_folder, target_date):
    """
    Find the NetCDF file that contains data for the target date.
    Files are named like MERIDA_TG_YYYYMM.nc
    """
    year_month = target_date.strftime("%Y%m")
    file_pattern = f"{data_folder}/MERIDA_TG_{year_month}.nc"
    matching_files = glob.glob(file_pattern)
    
    if matching_files:
        return matching_files[0]
    return None


def find_nearest_grid_point(target_lat, target_lon, grid_lat, grid_lon):
    """Find the nearest grid point to target coordinates."""
    lat_idx = np.argmin(np.abs(grid_lat - target_lat))
    lon_idx = np.argmin(np.abs(grid_lon - target_lon))
    return lat_idx, lon_idx


def get_ground_temperature_at_point(landslide_date, lat, lon, data_folder):
    """
    Get the most recent ground temperature value at a point.
    
    Parameters:
    - landslide_date: Datetime of the landslide event
    - lat, lon: Coordinates of the landslide
    - data_folder: Path to the folder containing NetCDF files
    
    Returns:
    - Ground temperature in Kelvin, or None if no data
    """
    target_date = pd.Timestamp(landslide_date)
    
    # Find the appropriate NetCDF file
    nc_file = find_nc_file_for_date(data_folder, target_date)
    
    if nc_file is None:
        return None
    
    try:
        dataset = nc.Dataset(nc_file, 'r')
        
        # Get coordinates
        grid_lat = dataset.variables['lat'][:]
        grid_lon = dataset.variables['lon'][:]
        
        # Find nearest grid point
        lat_idx, lon_idx = find_nearest_grid_point(lat, lon, grid_lat, grid_lon)
        
        # Get time data
        time_var = dataset.variables['time']
        time_units = time_var.units if hasattr(time_var, 'units') else 'days since 1900-01-01'
        times = nc.num2date(time_var[:], units=time_units)
        
        # Convert times to pandas Timestamps for comparison
        timestamps = []
        for t in times:
            if hasattr(t, 'year'):
                dt = datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
                timestamps.append(pd.Timestamp(dt))
            else:
                timestamps.append(pd.Timestamp(t))
        
        timestamps = np.array(timestamps)
        
        # Find the most recent timestamp that is <= landslide date
        valid_mask = timestamps <= target_date
        
        if not np.any(valid_mask):
            dataset.close()
            return None
        
        # Get the index of the most recent valid time
        valid_times = timestamps[valid_mask]
        time_idx = np.where(timestamps == valid_times[-1])[0][0]
        
        # Get ground temperature data (variable 'st' for surface temperature)
        st_data = dataset.variables['st']
        value = st_data[time_idx, lat_idx, lon_idx]
        
        dataset.close()
        
        # Check for missing/fill values
        if np.ma.is_masked(value) or np.isnan(value) or value < -1e30:
            return None
        
        return float(value)
        
    except Exception as e:
        print(f"Error reading {nc_file}: {e}")
        return None


def calculate_ground_temperature_for_landslides(landslides_df, data_folder):
    """
    Calculate ground temperature for all landslides.
    
    Parameters:
    - landslides_df: DataFrame with landslide data
    - data_folder: Path to folder containing NetCDF files
    
    Returns:
    - DataFrame with added ground temperature column
    """
    print("\nCalculating ground temperature for each landslide event...")
    
    result_df = landslides_df.copy()
    result_df['GroundTemperature_K'] = None
    
    # Process each landslide
    total = len(result_df)
    for idx, row in result_df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"  Processing landslide {idx + 1}/{total}...")
        
        date = row['Datum']
        lat = row['Latitude']
        lon = row['Longitude']
        
        if pd.isna(date):
            continue
        
        temp_value = get_ground_temperature_at_point(date, lat, lon, data_folder)
        result_df.at[idx, 'GroundTemperature_K'] = temp_value
    
    return result_df


def print_summary_statistics(df):
    """Print summary statistics for ground temperature data."""
    print("\n=== Summary Statistics ===")
    print(f"Total landslides processed: {len(df)}")
    
    col = 'GroundTemperature_K'
    valid_data = df[col].dropna()
    print(f"\nGround Temperature:")
    print(f"  Valid data points: {len(valid_data)}/{len(df)}")
    if len(valid_data) > 0:
        print(f"  Mean:   {valid_data.mean():.2f} K ({valid_data.mean() - 273.15:.2f} °C)")
        print(f"  Median: {valid_data.median():.2f} K ({valid_data.median() - 273.15:.2f} °C)")
        print(f"  Min:    {valid_data.min():.2f} K ({valid_data.min() - 273.15:.2f} °C)")
        print(f"  Max:    {valid_data.max():.2f} K ({valid_data.max() - 273.15:.2f} °C)")


def main():
    """Main function to process landslide and ground temperature data."""
    # Configuration
    shapefile_path = "Risikofaktoren/DataPreperation/create_shapes_and_test_data/output/Erdrutsche213_with_random.shp"
    data_folder = "Risikofaktoren/GroundTemperature/data"
    output_file = "Risikofaktoren/GroundTemperature/output/landslides_with_ground_temperature.csv"
    
    try:
        # Load landslide data
        landslides_df = load_landslide_data(shapefile_path)
        
        # Print first few rows
        print("\nFirst few landslide records:")
        print(landslides_df.head())
        # And the last few rows
        print("\nLast few landslide records:")
        print(landslides_df.tail())
        
        # Calculate ground temperature for all landslides
        result_df = calculate_ground_temperature_for_landslides(
            landslides_df, data_folder
        )
        
        # Save results
        result_df.to_csv(output_file, index=False)
        print(f"\n\nResults saved to {output_file}")
        
        # Display statistics
        print_summary_statistics(result_df)
        
        # Display sample of final results
        print("\n=== Sample Results (first 10 rows) ===")
        print(result_df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
