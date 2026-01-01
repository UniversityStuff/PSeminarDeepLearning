import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Pfad zum Raster anpassen


base_dir = Path("/Users/kiliandorn/Desktop/Universität/python/DeepAlpine")
raster_path = base_dir / "slope" / "slope_10m_UTM.tif"


# Raster laden


if not raster_path.exists():
    print(f"ERROR: Datei nicht gefunden: {raster_path}")
    exit(1)

from rasterio.enums import Resampling

# imports

with rasterio.open(raster_path) as src:
    # Prüfung auf Bildgröße und Downsampling falls nötig
    max_dim = 2000 # Maximale Breite/Höhe in Pixeln für den Plot
    h, w = src.height, src.width
    scale = min(1.0, max_dim / max(h, w))
    
    if scale < 1.0:
        new_h = int(h * scale)
        new_w = int(w * scale)
        print(f"WARNUNG: Raster ist groß ({h}x{w}). Downsampling auf {new_h}x{new_w}...")
        
        # 'out_shape' definiert die Zielgröße
        slope = src.read(
            1, 
            out_shape=(new_h, new_w),
            resampling=Resampling.bilinear,
            masked=True
        )
    else:
        slope = src.read(1, masked=True)

    bounds = src.bounds
    
    print(f"Raster geladen: {raster_path}")
    print(f"Shape im Speicher: {slope.shape}")
    print(f"CRS: {src.crs}")
    print(f"NoData-Wert im Metadaten: {src.nodata}")


# Berechnung nur auf validen Daten
valid_pixels = slope.count()
if valid_pixels == 0:
    print("WARNUNG: Alle Pixel sind maskiert (keine Daten)!")
    data_min = 0
    data_max = 1
else:
    # Statistik berechnen
    data_min = slope.min()
    data_max = slope.max()
    data_mean = slope.mean()
    
    # 2% und 98% Perzentile für besseren Kontrast berechnen
    # Dies ignoriert extreme Ausreißer und nutzt das Farbspektrum besser aus
    vmin = np.percentile(slope.compressed(), 2)
    vmax = np.percentile(slope.compressed(), 98)
    
    print(f"Gültige Pixel: {valid_pixels}")
    print(f"Original Range: Min={data_min:.2f}, Max={data_max:.2f}")
    print(f"Optimiertes Scaling (2%-98%): vmin={vmin:.2f}, vmax={vmax:.2f}")


# Plot


fig, ax = plt.subplots(figsize=(10, 8))

# Extent für korrekte Koordinaten-Achsen
left, bottom, right, top = bounds
extent = (left, right, bottom, top)

# Colormap: Grün (niedrig/flach) -> Rot (hoch/steil)
cmap = plt.cm.get_cmap("RdYlGn_r")
cmap.set_bad(color='white') # Maskierte Bereiche weiß

# Plot
im = ax.imshow(
    slope,
    cmap=cmap,
    vmin=vmin, 
    vmax=vmax,
    extent=extent,
    origin="upper"
)

ax.set_title("Slope Plot Südtirol")
ax.set_xlabel("Easting [UTM]")
ax.set_ylabel("Northing [UTM]")

# UTM-Koordinaten 
ax.ticklabel_format(style='plain', useOffset=False)
plt.xticks(rotation=45) 

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Slope [Degree]")

plt.tight_layout()

# Speichern als Bild
out_png = base_dir / "slope_suedtirol.png"
plt.savefig(out_png)
print(f"Fertiger Plot gespeichert unter: {out_png}")

plt.show()
