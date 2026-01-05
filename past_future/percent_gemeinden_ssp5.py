import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

gemeinden_shp_path = r"C:\Users\richt\Desktop\testNaturgefahrenOhneBuffer\prediction_modell\Gemeinden\FME_11060556_1767623643023_2240\Municipalities_polygon.shp" 
results_csv_path = r"C:\Users\richt\Desktop\testNaturgefahrenOhneBuffer\prediction_modell\RESULT_SSP585.csv"

gdf_gemeinden = gpd.read_file(gemeinden_shp_path)
df_points = pd.read_csv(results_csv_path)

gdf_points = gpd.GeoDataFrame(
    df_points,
    geometry=gpd.points_from_xy(df_points.Longitude, df_points.Latitude),
    crs="EPSG:4326"
)

if gdf_gemeinden.crs != gdf_points.crs:
    gdf_gemeinden = gdf_gemeinden.to_crs(gdf_points.crs)

gdf_points.geometry = gdf_points.geometry.buffer(0.02)

joined = gpd.sjoin(gdf_points, gdf_gemeinden, how="inner", predicate="intersects")

def percent_high_risk(series):
    return (series > 0.6).mean() * 100

gemeinde_risiko = joined.groupby('NAME_DE')['Landslide_Probability'].apply(percent_high_risk).reset_index()
gemeinde_risiko.rename(columns={'Landslide_Probability': 'High_Risk_Area_Percent'}, inplace=True)

final_map_data = gdf_gemeinden.merge(gemeinde_risiko, on='NAME_DE', how='left')

max_val = final_map_data['High_Risk_Area_Percent'].max()
print(f"HÖCHSTER WERT (Maximales Risiko einer Gemeinde): {max_val:.2f} %")

print("\nDie 10 gefährdetsten Gemeinden:")
print(final_map_data[['NAME_DE', 'High_Risk_Area_Percent']].sort_values(by='High_Risk_Area_Percent', ascending=False).head(10))

risk_col = 'High_Risk_Area_Percent'

# 1. Durchschnitt (Mean)
mean_val = final_map_data[risk_col].mean()
print(f"\nDurchschnittliches Risiko aller Gemeinden: {mean_val:.2f} %")

# 2. Median (aussagekräftiger bei Ausreißern)
median_val = final_map_data[risk_col].median()
print(f"Median-Risiko (die genaue Mitte): {median_val:.2f} %")

# 3. Wie viele sind komplett sicher? (0% Risiko)
safe_count = (final_map_data[risk_col] == 0).sum()
total_count = len(final_map_data)
print(f"Anzahl sicherer Gemeinden (0%): {safe_count} von {total_count}")

# 4. Wie viele sind extrem gefährdet? (> 50% der Fläche)
danger_count = (final_map_data[risk_col] > 50).sum()
print(f"Anzahl extrem gefährdeter Gemeinden (>50%): {danger_count}")

# 5. Verteilung (Quantile)
print("\nStatistische Verteilung:")
print(final_map_data[risk_col].describe())

fig, ax = plt.subplots(1, 1, figsize=(15, 12))
final_map_data.plot(
    column='High_Risk_Area_Percent', 
    ax=ax, 
    legend=True,
    cmap='RdYlGn_r',
    missing_kwds={'color': 'lightgrey', 'label': 'Keine Daten'},
    legend_kwds={'label': "Anteil Hochrisiko-Fläche % (SSP585)", 'shrink': 0.6},
    edgecolor='black',
    linewidth=0.5,
    vmin=0,
    vmax=100
)

plt.title("Risikokarte Südtiroler Gemeinden 2100 (SSP585)", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig('SSP585_Gemeinde_Risiko_Percent_Buffer.png', dpi=300)
plt.show()