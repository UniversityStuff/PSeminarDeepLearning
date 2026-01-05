import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# Config
train_file = 'landslides_training_data.csv'
shp_path = r"C:\Users\richt\Desktop\testNaturgefahrenOhneBuffer\prediction_modell\Gemeinden\FME_11060556_1767623643023_2240\Municipalities_polygon.shp"

# Features
features = [
    'Elevation', 'Slope', 'Aspect',
    'BIO01_Historical_Mean', 'BIO05_Historical_Max', 'BIO06_Historical_Min',
    'BIO12_Historical_Prec', 'BIO13_Historical_Prec', 'BIO15_Historical_Prec'
]

print("Lade Daten...")
df = pd.read_csv(train_file)

# Lat/Lon für die Karte später, also mitschleppen
X = df[features].fillna(0)
y = df['Target']
coords = df[['Latitude', 'Longitude']]

# Split
X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
    X, y, coords, test_size=0.2, random_state=42, stratify=y
)

print("Trainiere Modell für Analyse...")
rf = RandomForestClassifier(
    n_estimators=500, max_depth=10, min_samples_leaf=8, 
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

# Vorhersagen
y_pred = rf.predict(X_test)

# 1. METRIKEN TABELLE
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
}

df_metrics = pd.DataFrame([metrics])
df_metrics.to_csv('Model_Metrics_Table.csv', index=False)
print("\nMetriken gespeichert in 'Model_Metrics_Table.csv'")
print(df_metrics)

# 2. ERROR MAP (CONFUSION MAP)
# Kategorisieren der Vorhersagen
results = coords_test.copy()
results['Actual'] = y_test
results['Predicted'] = y_pred

def get_category(row):
    if row['Actual'] == 1 and row['Predicted'] == 1: return 'True Positive (Treffer)'
    if row['Actual'] == 0 and row['Predicted'] == 0: return 'True Negative (Korrekt Sicher)'
    if row['Actual'] == 0 and row['Predicted'] == 1: return 'False Positive (Fehlalarm)'
    if row['Actual'] == 1 and row['Predicted'] == 0: return 'False Negative (Verpasst)'

results['Category'] = results.apply(get_category, axis=1)

# Geometrie erstellen
gdf_results = gpd.GeoDataFrame(
    results, geometry=gpd.points_from_xy(results.Longitude, results.Latitude), crs="EPSG:4326"
)

# Shapefile für Hintergrund
gdf_gemeinden = gpd.read_file(shp_path)
if gdf_gemeinden.crs != gdf_results.crs:
    gdf_gemeinden = gdf_gemeinden.to_crs(gdf_results.crs)

# Plotten
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Hintergrund grau
gdf_gemeinden.plot(ax=ax, color='lightgrey', edgecolor='white', alpha=0.5)

# Punkte farblich codieren
# Farben: TP=grün, TN=blau, FP=Orange, FN=Rot
colors = {
    'True Positive (Treffer)': 'green',
    'True Negative (Korrekt Sicher)': 'blue',
    'False Positive (Fehlalarm)': 'orange',
    'False Negative (Verpasst)': 'red'
}

for cat, color in colors.items():
    subset = gdf_results[gdf_results['Category'] == cat]
    if len(subset) > 0:
        subset.plot(
            ax=ax, color=color, label=f"{cat} (n={len(subset)})", 
            markersize=40 if 'Negative' in cat else 60, # Fehler größer machen
            alpha=0.6 if 'True' in cat else 1.0, # Fehler deckend
            edgecolor='black'
        )

plt.title("Modell-Diagnose: Wo macht das Modell Fehler?", fontsize=16)
plt.legend()
plt.axis('off')
plt.tight_layout()
plt.savefig('Map_Error_Analysis.png', dpi=300)
plt.show()

print("Fertig. Karte gespeichert als 'Map_Error_Analysis.png'.")