import pandas as pd
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
# import seaborn as sns # Removed to avoid dependency issues
import numpy as np
import rasterio
from rasterio.enums import Resampling
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from pathlib import Path


# Config

base_dir = Path("/Users/kiliandorn/Desktop/Universität/python/DeepAlpine")

model_path = base_dir / "rf_landslides_beta.joblib"
val_csv_path = base_dir / "landslides_validation_data_beta.csv"
raster_path = base_dir / "slope" / "slope_10m_UTM.tif"
boundary_shp_path = base_dir / "Untersuchungsgebiet" / "Untersuchungsgebiet_32632.shp"

# Output paths
cm_plot_path = base_dir / "confusion_matrix_beta.png"
map_plot_path = base_dir / "prediction_map_beta.png"

# =====================================================================
# 1. Load Data & Model
# =====================================================================

print(f"Loading Model: {model_path}")
clf = joblib.load(model_path)

print(f"Loading Validation Data: {val_csv_path}")
df_val = pd.read_csv(val_csv_path)

# Prepare Features (exclude non-feature columns)
target_column = "hazard_binary"
drop_columns = ["Datum", "UUID", "MOVEMENT_C", "MOVEMENT_D", target_column, "Latitude", "Longitude", "geometry"]
existing_drop = [c for c in drop_columns if c in df_val.columns]

X_val = df_val.drop(columns=existing_drop)
y_true = df_val[target_column]

# =====================================================================
# 2. Prediction
# =====================================================================

print("Running predictions...")
# Prediction mit Wahrscheinlichkeiten
y_proba = clf.predict_proba(X_val)
class_labels = clf.classes_
hazard_idx = list(class_labels).index("hazard")

hazard_proba = y_proba[:, hazard_idx]

# Eigene Schwelle definieren – z.B. 0.4 statt 0.5
threshold = 0.3
y_pred = np.where(hazard_proba >= threshold, "hazard", "no_hazard")

# Add prediction to dataframe for plotting logic
df_val["Prediction"] = y_pred
df_val["Correct"] = (df_val["Prediction"] == y_true)

# Print Report
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred))

# Calculate Outcome Statistics
n_total = len(df_val)
n_correct = df_val["Correct"].sum()
n_wrong = n_total - n_correct
acc = (n_correct / n_total) * 100

print(f"\n=== Prediction Outcome Summary ===")
print(f"Total Validation Samples: {n_total}")
print(f"Correct Predictions:      {n_correct} ({acc:.2f}%)")
print(f"Wrong Predictions:        {n_wrong} ({100-acc:.2f}%)")

# Specific "Landslide" Detection (ignoring "Keine Rutschung")
# We assume the negative class contains "Keine" or "No".
# Adjust this string if your exact label is different (e.g., "No Movement")
neg_class_keyword = "no_hazard" 

# Filter for True Landslides (Actual = Landslide)
landslides_only = df_val[~df_val[target_column].astype(str).str.contains(neg_class_keyword, case=False, na=False)]
nm_total = len(landslides_only)

if nm_total > 0:
    nm_correct = landslides_only["Correct"].sum()
    nm_acc = (nm_correct / nm_total) * 100
    print(f"\n=== LANDSLIDE Detection Summary (excluding '{neg_class_keyword}...') ===")
    print(f"Total Landslide Events:   {nm_total}")
    print(f"Detected Correctly:       {nm_correct} ({nm_acc:.2f}%)")
    print(f"Missed (False Negative):  {nm_total - nm_correct} ({100-nm_acc:.2f}%)")
else:
    print(f"\nNo Landslide events found (checked for classes without '{neg_class_keyword}').")

# =====================================================================
# 3. Plot Confusion Matrix
# =====================================================================

print("Generating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred, labels=clf.classes_)

# Use sklearn's built-in display or matplotlib directly
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)

plt.title("Confusion Matrix (Validation Data)")
plt.tight_layout()
plt.savefig(cm_plot_path)
print(f"Saved Confusion Matrix to: {cm_plot_path}")
plt.close()

# =====================================================================
# 4. Plot Prediction Map
# =====================================================================

print("Generating Prediction Map...")

# --- 4a. Load Slope Raster (Background) ---
# We use the robust logic from plot_slope.py (downsampling)
try:
    with rasterio.open(raster_path) as src:
        max_dim = 2000
        h, w = src.height, src.width
        scale = min(1.0, max_dim / max(h, w))
        
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            slope = src.read(1, out_shape=(new_h, new_w), resampling=Resampling.bilinear, masked=True)
            print(f"Downsampled raster to {new_h}x{new_w} for background.")
        else:
            slope = src.read(1, masked=True)
            
        bounds = src.bounds
        crs = src.crs
except Exception as e:
    print(f"Error loading raster: {e}")
    exit(1)

# --- 4b. Prepare Points ---
gdf = gpd.GeoDataFrame(
    df_val,
    geometry=gpd.points_from_xy(df_val.Longitude, df_val.Latitude),
    crs="EPSG:4326"
)

if crs is not None and gdf.crs != crs:
    print(f"Reprojecting points to {crs}...")
    gdf = gdf.to_crs(crs)

# --- 4c. Plotting ---
# --- 4c. Plotting ---
fig, ax = plt.subplots(figsize=(12, 10))

# Extent
left, bottom, right, top = bounds
extent = (left, right, bottom, top)

# Background: Region Outline (from raster mask)
if hasattr(slope, "mask"):
    valid_mask = ~np.ma.getmaskarray(slope)
    valid_mask = valid_mask.astype(int)
    ax.contour(
        valid_mask, 
        levels=[0.5], 
        colors='black', 
        linewidths=1, 
        extent=extent,
        origin='upper'
    )
else:
    rect = plt.Rectangle((left, bottom), right-left, top-bottom, 
                         fill=False, edgecolor='black', linewidth=1)
    ax.add_patch(rect)

# Untersuchungsgebiet boundary as dashed line
try:
    boundary_gdf = gpd.read_file(boundary_shp_path)
    if crs is not None and boundary_gdf.crs != crs:
        boundary_gdf = boundary_gdf.to_crs(crs)
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=2, linestyle='--', zorder=2)
    print(f"Loaded Untersuchungsgebiet boundary from: {boundary_shp_path}")
except Exception as e:
    print(f"Warning: Could not load boundary shapefile: {e}")

# Define Categories
# Wahrer Zustand (Ground Truth)
# target_column ist "hazard_binary"
is_true_hazard = df_val[target_column] == "hazard"

# Vorhersage des Modells
is_pred_hazard = df_val["Prediction"] == "hazard"

# TN / TP / FN / FP definieren
tn_mask = (~is_true_hazard) & (~is_pred_hazard)  # echte no_hazard, korrekt als no_hazard vorhergesagt
tp_mask = (is_true_hazard) & (is_pred_hazard)    # echte hazard, korrekt als hazard vorhergesagt
fn_mask = (is_true_hazard) & (~is_pred_hazard)   # echte hazard, aber als no_hazard vorhergesagt
fp_mask = (~is_true_hazard) & (is_pred_hazard)   # echte no_hazard, aber als hazard vorhergesagt

tn_points = gdf[tn_mask]
tp_points = gdf[tp_mask]
fn_points = gdf[fn_mask]
fp_points = gdf[fp_mask]

print(f"Plotting details:")
print(f"  TN (Correct No-Slide): {len(tn_points)} -> Orange")
print(f"  TP (Correct Slide):    {len(tp_points)} -> Green (Star)")
print(f"  FN (Missed Slide):     {len(fn_points)} -> Red (Cross)")
print(f"  FP (False Alarm):      {len(fp_points)} -> Purple")


# Plot TN (Yellow, s=20)
ax.scatter(
    tn_points.geometry.x, tn_points.geometry.y,
    c='orange', s=20, alpha=0.6, label='Correct No-Slide (Orange)'
)

# Plot FP (Purple, s=40)
ax.scatter(
    fp_points.geometry.x, fp_points.geometry.y,
    c='purple', marker='o', s=40, label='False Alarm (Purple)'
)

# Plot TP (Green, Star, Highlight)
ax.scatter(
    tp_points.geometry.x, tp_points.geometry.y,
    c='green',          # Green
    edgecolor='black',  # Outline
    marker='*', 
    s=150,              # Big size
    label='Correct Landslide (Green)',
    zorder=3
)

# Plot FN (Red, X, Highlight)
ax.scatter(
    fn_points.geometry.x, fn_points.geometry.y,
    c='red',            # Red
    edgecolor='black',
    marker='x',         # Red Cross
    linewidths=2,
    s=100,              # Big size
    label='Missed Landslide (Red Cross)',
    zorder=3
)

ax.set_title("Landslide Prediction Map: Analysis")
ax.set_xlabel("Easting [UTM]")
ax.set_ylabel("Northing [UTM]")
ax.ticklabel_format(style='plain', useOffset=False)
plt.xticks(rotation=45)
ax.set_aspect('equal')

# Legend
ax.legend(loc='upper right')

# Add Stats Text Box
stats_text = (
    f"TN (Correct No-Slide): {len(tn_points)}\n"
    f"TP (Correct Slide):    {len(tp_points)}\n"
    f"FN (Missed Slide):     {len(fn_points)}\n"
    f"FP (False Alarm):      {len(fp_points)}"
)
# Place text box in lower left in axes coordinates
ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(map_plot_path, dpi=300)
print(f"Saved Prediction Map to: {map_plot_path}")
plt.show()
