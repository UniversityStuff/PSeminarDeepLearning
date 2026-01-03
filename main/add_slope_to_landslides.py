import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.windows import Window
from pathlib import Path

# =====================================================================
# Config
# =====================================================================

base_dir = Path("/Users/kiliandorn/Desktop/Universität/python/DeepAlpine")
csv_path = base_dir / "landslides_Rain_SM.csv"
raster_path = base_dir / "slope" / "slope_10m_UTM.tif"
out_csv = base_dir / "landslides_Rain_SM_with_slope_AntiGravity.csv"

# =====================================================================
# 1. Load Data
# =====================================================================

print(f"Loading CSV: {csv_path}")
df = pd.read_csv(csv_path)
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
    crs="EPSG:4326"
)

# =====================================================================
# 2. Extract Slope Features
# =====================================================================

slope_pixel = []
slope_mean_3x3 = []
slope_max_3x3 = []

with rasterio.open(raster_path) as src:
    print(f"Raster loaded: {raster_path} ({src.crs})")
    
    # Reproject points if needed
    if gdf.crs != src.crs:
        print(f"Reprojecting points from {gdf.crs} to {src.crs}...")
        gdf = gdf.to_crs(src.crs)

    print("Extracting features for", len(gdf), "points...")
    
    # Get affine transform for pixel conversion
    inv_transform = ~src.transform

    for idx, row in gdf.iterrows():
        # Get pixel coordinates
        x, y = row.geometry.x, row.geometry.y
        px, py = inv_transform * (x, y)
        px, py = int(px), int(py)

        # 1. Exact Pixel Value (Point Sampling)
        # We use a 1x1 window reading to be safe/efficient loop-wise, 
        # or just simple array indexing if loaded fully (but raster is too big).
        # Efficient way: Read small window around point.
        
        # Window: 3x3 centered on pixel
        # px-1 to px+2 (exclusive) -> 3 pixels wide
        
        try:
            window = Window(px - 1, py - 1, 3, 3)
            
            # Clip window to raster size
            window = window.intersection(Window(0, 0, src.width, src.height))
            
            data = src.read(1, window=window)
            
            if data.size == 0:
                slope_pixel.append(np.nan)
                slope_mean_3x3.append(np.nan)
                slope_max_3x3.append(np.nan)
                continue
                
            # Handle NoData
            if src.nodata is not None:
                data = data.astype(float)
                data[data == src.nodata] = np.nan

            # Central pixel is at index [1, 1] if window is full 3x3
            # If clipped, we need to be careful, but mean/max are robust.
            # Only exact pixel needs care. 
            
            # Simplified: Use sample() for exact pixel
            gen = src.sample([(x, y)])
            val = next(gen)[0]
            if src.nodata is not None and val == src.nodata:
                val = np.nan
            slope_pixel.append(val)
            
            # Neighborhood stats
            if np.all(np.isnan(data)):
                slope_mean_3x3.append(np.nan)
                slope_max_3x3.append(np.nan)
            else:
                slope_mean_3x3.append(np.nanmean(data))
                slope_max_3x3.append(np.nanmax(data))
                
        except Exception as e:
            print(f"Error at point {idx}: {e}")
            slope_pixel.append(np.nan)
            slope_mean_3x3.append(np.nan)
            slope_max_3x3.append(np.nan)

# =====================================================================
# 3. Save Result
# =====================================================================

df["Slope"] = slope_pixel
df["Slope_Mean_30m"] = slope_mean_3x3
df["Slope_Max_30m"] = slope_max_3x3

print("\n--- Summary Statistics ---")
print(df[["Slope", "Slope_Mean_30m", "Slope_Max_30m"]].describe())

df.to_csv(out_csv, index=False)
print(f"\nSaved updated CSV to: {out_csv}")
