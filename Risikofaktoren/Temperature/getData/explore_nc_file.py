import netCDF4 as nc
import numpy as np


""" Just a simple script for showing the contents of a NetCDF file"""

# Open the NetCDF file
file_path = "Risikofaktoren/Temperature/data/MERIDA_TT_202206.nc"
dataset = nc.Dataset(file_path, 'r')

print("=" * 60)
print(f"NetCDF File: {file_path}")
print("=" * 60)

# Display file dimensions
print("\nDimensions:")
for dim_name, dim in dataset.dimensions.items():
    print(f"  {dim_name}: {len(dim)} {'(unlimited)' if dim.isunlimited() else ''}")

# Display variables
print("\nVariables:")
for var_name, var in dataset.variables.items():
    print(f"\n  {var_name}:")
    print(f"    Shape: {var.shape}")
    print(f"    Dimensions: {var.dimensions}")
    print(f"    Data type: {var.dtype}")
    
    # Display attributes
    if var.ncattrs():
        print(f"    Attributes:")
        for attr in var.ncattrs():
            print(f"      {attr}: {var.getncattr(attr)}")
    
    # Show some sample data for small variables
    if var.size < 100:
        print(f"    Data: {var[:]}")
    else:
        print(f"    Data preview (first few values): {var[:].flat[:5]}...")

# Display global attributes
print("\nGlobal Attributes:")
for attr in dataset.ncattrs():
    print(f"  {attr}: {dataset.getncattr(attr)}")

# Close the dataset
dataset.close()

print("\n" + "=" * 60)
