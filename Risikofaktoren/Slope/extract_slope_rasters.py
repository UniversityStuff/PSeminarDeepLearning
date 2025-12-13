"""
Extract slope raster data for each 5x5km polygon.

This script clips the slope raster for each polygon and saves individual 
raster files. Polygons with more than 5% NoData are excluded.
"""

import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import os
from pathlib import Path
import json


def load_data(raster_path, polygon_shapefile_path):
    """Load the slope raster and polygon shapefile."""
    print("Loading slope raster...")
    raster = rasterio.open(raster_path)
    
    print("Loading polygons...")
    gdf = gpd.read_file(polygon_shapefile_path)
    print(f"Loaded {len(gdf)} polygons, CRS: {gdf.crs}")
    
    # Transform to raster CRS if needed
    if gdf.crs != raster.crs:
        print(f"Reprojecting from {gdf.crs} to {raster.crs}")
        gdf = gdf.to_crs(raster.crs)
    
    return raster, gdf


def calculate_nodata_percentage(data, nodata_value):
    """Calculate the percentage of NoData values in the array."""
    if nodata_value is None:
        # If no nodata value defined, check for very negative values (common for float rasters)
        nodata_mask = data < -1e30
    else:
        nodata_mask = np.isclose(data, nodata_value, rtol=1e-5) | (data < -1e30)
    
    total_pixels = data.size
    nodata_pixels = np.sum(nodata_mask)
    
    return (nodata_pixels / total_pixels) * 100 if total_pixels > 0 else 100


def extract_raster_for_polygon(raster, polygon_geometry, nodata_value):
    """
    Extract raster data clipped to a polygon.
    
    Returns:
        tuple: (clipped_data, transform, nodata_percentage) or (None, None, nodata_pct) if invalid
    """
    try:
        # Clip raster to polygon
        out_image, out_transform = mask(raster, [polygon_geometry], crop=True, 
                                         nodata=nodata_value, filled=True)
        
        # Calculate NoData percentage
        nodata_pct = calculate_nodata_percentage(out_image[0], nodata_value)
        
        return out_image[0], out_transform, nodata_pct
        
    except Exception as e:
        print(f"  Error clipping: {e}")
        return None, None, 100.0


def save_raster(output_path, data, transform, crs, nodata_value):
    """Save a raster array to a GeoTIFF file."""
    profile = {
        'driver': 'GTiff',
        'dtype': data.dtype,
        'width': data.shape[1],
        'height': data.shape[0],
        'count': 1,
        'crs': crs,
        'transform': transform,
        'nodata': nodata_value,
        'compress': 'lzw'
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data, 1)


def extract_all_slope_rasters(raster, gdf, output_dir, max_nodata_pct=5.0):
    """
    Extract slope raster for each polygon and save to individual files.
    
    Parameters:
        raster: Open rasterio dataset
        gdf: GeoDataFrame with polygons
        output_dir: Directory to save output rasters
        max_nodata_pct: Maximum allowed NoData percentage (default 5%)
    
    Returns:
        dict: Summary of extraction results
    """
    # Create output directories
    valid_dir = Path(output_dir) / "valid_rasters"
    valid_dir.mkdir(parents=True, exist_ok=True)
    
    nodata_value = raster.nodata
    results = {
        'total': len(gdf),
        'valid': 0,
        'excluded_nodata': 0,
        'excluded_error': 0,
        'valid_uuids': [],
        'excluded_uuids': []
    }
    
    print(f"\nExtracting slope rasters for {len(gdf)} polygons...")
    print(f"Maximum allowed NoData: {max_nodata_pct}%")
    print("-" * 60)
    
    for idx, row in gdf.iterrows():
        uuid = row['UUID']
        is_landslide = row['MOVEMENT_C'] != 99
        category = "landslide" if is_landslide else "no_landslide"
        
        # Progress indicator
        if (idx + 1) % 50 == 0:
            print(f"Processing {idx + 1}/{len(gdf)}...")
        
        # Extract raster for this polygon
        data, transform, nodata_pct = extract_raster_for_polygon(
            raster, row.geometry, nodata_value
        )
        
        if data is None:
            results['excluded_error'] += 1
            results['excluded_uuids'].append({'uuid': uuid, 'reason': 'extraction_error'})
            continue
        
        if nodata_pct > max_nodata_pct:
            results['excluded_nodata'] += 1
            results['excluded_uuids'].append({
                'uuid': uuid, 
                'reason': 'nodata_exceeded',
                'nodata_pct': round(nodata_pct, 2)
            })
            continue
        
        # Save valid raster
        output_filename = f"{category}_{uuid}.tif"
        output_path = valid_dir / output_filename
        
        save_raster(output_path, data, transform, raster.crs, nodata_value)
        
        results['valid'] += 1
        results['valid_uuids'].append({
            'uuid': uuid,
            'category': category,
            'nodata_pct': round(nodata_pct, 2),
            'shape': data.shape,
            'filename': output_filename
        })
    
    return results


def print_summary(results):
    """Print summary of extraction results."""
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total polygons processed: {results['total']}")
    print(f"Valid rasters saved:      {results['valid']}")
    print(f"Excluded (>5% NoData):    {results['excluded_nodata']}")
    print(f"Excluded (errors):        {results['excluded_error']}")
    print(f"Success rate:             {results['valid']/results['total']*100:.1f}%")
    
    # Count by category
    landslides = sum(1 for r in results['valid_uuids'] if r['category'] == 'landslide')
    non_landslides = results['valid'] - landslides
    print(f"\nValid landslide rasters:     {landslides}")
    print(f"Valid non-landslide rasters: {non_landslides}")


def main():
    """Main function to extract slope rasters."""
    # Paths
    raster_path = "Risikofaktoren/Slope/Input/slope_10m_UTM.tif"
    polygon_shapefile_path = "Risikofaktoren/DataPreperation/create_shapes_and_test_data/output/Erdrutsche213_with_random_5x5km.shp"
    output_dir = "Risikofaktoren/Slope/Output"
    
    # Load data
    raster, gdf = load_data(raster_path, polygon_shapefile_path)
    
    # Extract rasters
    results = extract_all_slope_rasters(raster, gdf, output_dir, max_nodata_pct=5.0)
    
    # Print summary
    print_summary(results)
    
    # Save results metadata
    metadata_path = Path(output_dir) / "extraction_results.json"
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Close raster
    raster.close()
    print("\nExtraction complete!")


if __name__ == "__main__":
    main()
