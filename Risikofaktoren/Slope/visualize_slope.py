"""
Visualize slope data and landslide points.

This script loads the slope raster and overlays the landslide/non-landslide points
to provide an initial visualization of the terrain data.
"""

import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer


def load_slope_raster(raster_path):
    """Load the slope raster and return the dataset and data array."""
    dataset = rasterio.open(raster_path)
    slope_data = dataset.read(1)
    return dataset, slope_data


def load_landslide_points(shapefile_path):
    """Load landslide points from shapefile."""
    gdf = gpd.read_file(shapefile_path)
    print(f"Loaded {len(gdf)} points")
    print(f"CRS: {gdf.crs}")
    return gdf


def transform_points_to_raster_crs(gdf, raster_crs):
    """Transform points to match raster CRS if needed."""
    if gdf.crs != raster_crs:
        print(f"Transforming from {gdf.crs} to {raster_crs}")
        gdf = gdf.to_crs(raster_crs)
    return gdf


def visualize_slope_overview(slope_dataset, slope_data, title="Slope Map (degrees)"):
    """Create an overview visualization of the full slope raster."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a custom colormap for slope visualization
    # Low slopes (flat) = green, medium = yellow, high (steep) = red/brown
    colors = ['#2d5016', '#4a7c23', '#8bc34a', '#cddc39', '#ffeb3b', 
              '#ff9800', '#ff5722', '#bf360c', '#4e342e']
    cmap = LinearSegmentedColormap.from_list('slope', colors, N=256)
    
    # Mask NoData values
    slope_masked = np.ma.masked_where(slope_data < 0, slope_data)
    
    # Get extent for proper geographic display
    extent = [slope_dataset.bounds.left, slope_dataset.bounds.right,
              slope_dataset.bounds.bottom, slope_dataset.bounds.top]
    
    im = ax.imshow(slope_masked, cmap=cmap, extent=extent, 
                   vmin=0, vmax=60, interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax, label='Slope (degrees)', shrink=0.8)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    
    plt.tight_layout()
    return fig, ax


def visualize_slope_with_points(slope_dataset, slope_data, gdf):
    """Visualize slope raster with landslide and non-landslide points overlaid."""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Custom colormap for slope
    colors = ['#2d5016', '#4a7c23', '#8bc34a', '#cddc39', '#ffeb3b', 
              '#ff9800', '#ff5722', '#bf360c', '#4e342e']
    cmap = LinearSegmentedColormap.from_list('slope', colors, N=256)
    
    # Mask NoData values
    slope_masked = np.ma.masked_where(slope_data < 0, slope_data)
    
    # Get extent
    extent = [slope_dataset.bounds.left, slope_dataset.bounds.right,
              slope_dataset.bounds.bottom, slope_dataset.bounds.top]
    
    im = ax.imshow(slope_masked, cmap=cmap, extent=extent, 
                   vmin=0, vmax=60, interpolation='nearest', alpha=0.9)
    
    # Separate landslide and non-landslide points
    landslides = gdf[gdf['MOVEMENT_C'] != 99]
    non_landslides = gdf[gdf['MOVEMENT_C'] == 99]
    
    print(f"Landslide points: {len(landslides)}")
    print(f"Non-landslide (random) points: {len(non_landslides)}")
    
    # Plot points
    if len(non_landslides) > 0:
        non_landslides.plot(ax=ax, color='blue', markersize=15, alpha=0.6, 
                            marker='o', label=f'Non-landslide ({len(non_landslides)})')
    
    if len(landslides) > 0:
        landslides.plot(ax=ax, color='red', markersize=25, alpha=0.8, 
                        marker='^', edgecolor='white', linewidth=0.5,
                        label=f'Landslide ({len(landslides)})')
    
    cbar = plt.colorbar(im, ax=ax, label='Slope (degrees)', shrink=0.7)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Slope Map with Landslide Locations\n(South Tyrol, Italy)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    
    plt.tight_layout()
    return fig, ax


def visualize_slope_with_polygons(slope_dataset, slope_data, gdf_points, gdf_polygons):
    """Visualize slope raster with 5x5km polygons and points overlaid."""
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Custom colormap for slope
    colors = ['#2d5016', '#4a7c23', '#8bc34a', '#cddc39', '#ffeb3b', 
              '#ff9800', '#ff5722', '#bf360c', '#4e342e']
    cmap = LinearSegmentedColormap.from_list('slope', colors, N=256)
    
    # Mask NoData values
    slope_masked = np.ma.masked_where(slope_data < 0, slope_data)
    
    # Get extent
    extent = [slope_dataset.bounds.left, slope_dataset.bounds.right,
              slope_dataset.bounds.bottom, slope_dataset.bounds.top]
    
    im = ax.imshow(slope_masked, cmap=cmap, extent=extent, 
                   vmin=0, vmax=60, interpolation='nearest', alpha=0.85)
    
    # Separate landslide and non-landslide data
    landslide_polys = gdf_polygons[gdf_polygons['MOVEMENT_C'] != 99]
    non_landslide_polys = gdf_polygons[gdf_polygons['MOVEMENT_C'] == 99]
    landslide_pts = gdf_points[gdf_points['MOVEMENT_C'] != 99]
    non_landslide_pts = gdf_points[gdf_points['MOVEMENT_C'] == 99]
    
    print(f"Landslide polygons: {len(landslide_polys)}")
    print(f"Non-landslide polygons: {len(non_landslide_polys)}")
    
    # Plot polygons (boundaries only, no fill)
    if len(non_landslide_polys) > 0:
        non_landslide_polys.boundary.plot(ax=ax, color='cyan', linewidth=1.0, 
                                           alpha=0.7, linestyle='--')
    
    if len(landslide_polys) > 0:
        landslide_polys.boundary.plot(ax=ax, color='magenta', linewidth=1.5, 
                                       alpha=0.9, linestyle='-')
    
    # Plot center points
    if len(non_landslide_pts) > 0:
        non_landslide_pts.plot(ax=ax, color='cyan', markersize=10, alpha=0.7, 
                               marker='o', edgecolor='darkblue', linewidth=0.3,
                               label=f'Non-landslide 5×5km ({len(non_landslide_pts)})')
    
    if len(landslide_pts) > 0:
        landslide_pts.plot(ax=ax, color='magenta', markersize=20, alpha=0.9, 
                           marker='^', edgecolor='white', linewidth=0.5,
                           label=f'Landslide 5×5km ({len(landslide_pts)})')
    
    cbar = plt.colorbar(im, ax=ax, label='Slope (degrees)', shrink=0.7)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_title('Slope Map with 5×5km Analysis Polygons\n(South Tyrol, Italy)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    
    plt.tight_layout()
    return fig, ax


def visualize_slope_histogram(slope_data):
    """Create a histogram of slope values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter valid values
    valid_slopes = slope_data[(slope_data >= 0) & (slope_data <= 90)]
    
    ax.hist(valid_slopes.flatten(), bins=60, color='steelblue', 
            edgecolor='navy', alpha=0.7)
    ax.set_xlabel('Slope (degrees)', fontsize=12)
    ax.set_ylabel('Pixel Count', fontsize=12)
    ax.set_title('Distribution of Slope Values', fontsize=14, fontweight='bold')
    ax.axvline(x=np.median(valid_slopes), color='red', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(valid_slopes):.1f}°')
    ax.axvline(x=np.mean(valid_slopes), color='orange', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(valid_slopes):.1f}°')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def print_raster_info(slope_dataset, slope_data):
    """Print information about the slope raster."""
    print("\n=== Slope Raster Information ===")
    print(f"Shape: {slope_data.shape}")
    print(f"CRS: {slope_dataset.crs}")
    print(f"Resolution: {slope_dataset.res[0]} x {slope_dataset.res[1]} meters")
    print(f"Bounds: {slope_dataset.bounds}")
    print(f"No Data Value: {slope_dataset.nodata}")
    
    valid_data = slope_data[(slope_data >= 0) & (slope_data <= 90)]
    print(f"\nSlope Statistics (valid values only):")
    print(f"  Min: {valid_data.min():.2f}°")
    print(f"  Max: {valid_data.max():.2f}°")
    print(f"  Mean: {valid_data.mean():.2f}°")
    print(f"  Median: {np.median(valid_data):.2f}°")
    print(f"  Std Dev: {valid_data.std():.2f}°")


def main():
    """Main function to visualize slope data."""
    # Paths
    raster_path = "Risikofaktoren/Slope/Input/slope_10m_UTM.tif"
    shapefile_path = "Risikofaktoren/DataPreperation/create_shapes_and_test_data/output/Erdrutsche213_with_random.shp"
    polygon_shapefile_path = "Risikofaktoren/DataPreperation/create_shapes_and_test_data/output/Erdrutsche213_with_random_5x5km.shp"
    output_dir = "Risikofaktoren/Slope/Output"
    
    # Ensure output directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading slope raster...")
    slope_dataset, slope_data = load_slope_raster(raster_path)
    print_raster_info(slope_dataset, slope_data)
    
    print("\nLoading landslide points...")
    gdf = load_landslide_points(shapefile_path)
    
    print("\nLoading 5x5km polygons...")
    gdf_polygons = load_landslide_points(polygon_shapefile_path)
    
    # Transform points and polygons to raster CRS
    gdf_transformed = transform_points_to_raster_crs(gdf, slope_dataset.crs)
    gdf_polygons_transformed = transform_points_to_raster_crs(gdf_polygons, slope_dataset.crs)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Overview map
    fig1, ax1 = visualize_slope_overview(slope_dataset, slope_data)
    fig1.savefig(f"{output_dir}/slope_overview.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/slope_overview.png")
    
    # 2. Map with points
    fig2, ax2 = visualize_slope_with_points(slope_dataset, slope_data, gdf_transformed)
    fig2.savefig(f"{output_dir}/slope_with_points.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/slope_with_points.png")
    
    # 3. Map with 5x5km polygons
    fig3, ax3 = visualize_slope_with_polygons(slope_dataset, slope_data, 
                                               gdf_transformed, gdf_polygons_transformed)
    fig3.savefig(f"{output_dir}/slope_with_polygons.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/slope_with_polygons.png")
    
    # 4. Histogram
    fig4, ax4 = visualize_slope_histogram(slope_data)
    fig4.savefig(f"{output_dir}/slope_histogram.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/slope_histogram.png")
    
    # Close the raster dataset
    slope_dataset.close()
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
