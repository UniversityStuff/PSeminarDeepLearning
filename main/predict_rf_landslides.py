import pandas as pd
import joblib
import numpy as np

# Pfade anpassen (Nutzer-Definitionen beibehalten)
csv_path = "/Users/kiliandorn/Desktop/Universität/python/DeepAlpine/landslides_validation_data_beta.csv"
model_path = "/Users/kiliandorn/Desktop/Universität/python/DeepAlpine/rf_landslides_beta.joblib"
output_path = "/Users/kiliandorn/Desktop/Universität/python/DeepAlpine/landslides_preds_beta.csv"

# Daten einlesen
print(f"Lade Daten von: {csv_path}")
df = pd.read_csv(csv_path)

# Modell laden
print(f"Lade Modell von: {model_path}")
clf = joblib.load(model_path)

# Features vorbereiten (GENAU wie im Training!)
target_column = "hazard_binary"
# WICHTIG: Koordinaten, Geometrie und ursprüngliche Klassen müssen raus
drop_columns = ["Datum", "UUID", "MOVEMENT_C", "MOVEMENT_D", target_column, "Latitude", "Longitude", "geometry"]

# Nur Spalten droppen, die auch wirklich da sind
existing_drop = [c for c in drop_columns if c in df.columns]
X = df.drop(columns=existing_drop)

print("Features für Vorhersage:")
print(X.columns.tolist())

# Vorhersagen
print("Starte Vorhersage...")
y_pred = clf.predict(X)
y_proba = clf.predict_proba(X)

# Ergebnisse anfügen
df["predicted_class"] = y_pred

# Wahrscheinlichkeiten für jede Klasse hinzufügen
class_labels = clf.named_steps["model"].classes_
for i, cls in enumerate(class_labels):
    df[f"proba_{cls}"] = y_proba[:, i]

# Risiko-Analyse (Optional: "Better Outcome")
# Annahme: Es gibt eine Klasse, die für "Rutschung" steht (nicht "no movement" o.ä.)
# Wir suchen die Klasse mit der höchsten Wahrscheinlichkeit, die NICHT "keine" ist (falls vorhanden).
# Hier vereinfacht: Wir definieren "Confidence" als die Wahrscheinlichkeit der vorhergesagten Klasse.
df["confidence"] = np.max(y_proba, axis=1)

# Risk Level basierend auf Confidence (nur ein Beispiel)
# High Risk = > 80% sicher
# Medium Risk = 50%
def get_risk_level(row):
    conf = row["confidence"]
    if conf > 0.8:
        return "High"
    elif conf > 0.5:
        return "Medium"
    else:
        return "Low"

df["risk_level"] = df.apply(get_risk_level, axis=1)

# Speichern
df.to_csv(output_path, index=False)
print(f"Vorhersagen gespeichert in: {output_path}")

print("\n--- Zusammenfassung ---")
print(df["predicted_class"].value_counts())
