import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, precision_score
import joblib

# CSV einlesen
csv_path = "/Users/kiliandorn/Desktop/Universität/python/DeepAlpine/landslides_beta.csv"
df = pd.read_csv(csv_path)

print("Erste Zeilen der Tabelle:")
print(df.head())
print("\nSpalten:")
print(df.columns)

# Zielvariable & Features

# alle Ereignisklassen
hazard_classes = [
    "Gleitung rotational/translational",
    "Schnelles Fliessen (Mure)",
    "Komplexe Rutschung",
]

df["hazard_binary"] = df["MOVEMENT_D"].apply(
    lambda c: "hazard" if c in hazard_classes else "no_hazard"
)

target_column = "hazard_binary"
drop_columns = ["Datum", "UUID", "MOVEMENT_C", "MOVEMENT_D", target_column, "Latitude", "Longitude", "geometry"]
existing_drop = [c for c in drop_columns if c in df.columns]

X_all = df.drop(columns=existing_drop)
y_all = df[target_column]

print("\nFeature-Spalten:")
print(X_all.columns)

print("\nVerteilung der Zielklassen (Gesamt):")
print(y_all.value_counts())


# 70/30 Split (Train / Validation-for-later)


# Stratify ist wichtig, damit Klassenverhältnisse erhalten bleiben
X_train, X_val_for_later, y_train, y_val_for_later = train_test_split(
    X_all, y_all,
    test_size=0.3,          # oder 0.2
    stratify=y_all,
    random_state=42
)

# Indizes nutzen, um die Original-Zeilen für die CSV zu retten
train_indices = X_train.index
val_indices = X_val_for_later.index

df_val_export = df.loc[val_indices]
val_csv_path = "/Users/kiliandorn/Desktop/Universität/python/DeepAlpine/landslides_validation_data_beta.csv"
df_val_export.to_csv(val_csv_path, index=False)
print(f"\n50% Validation-Daten gespeichert unter:\n{val_csv_path}")

print(f"\nTrainingsdaten: {X_train.shape}")
print(f"Validation-Daten (nicht für Training genutzt): {X_val_for_later.shape}")

# Alle Features sind numerisch
numeric_features = X_train.columns.tolist()

# Preprocessing & Modell Pipeline


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
    ]
)


# Modell-Optimierung (GridSearch)


print("\nStarte Hyperparameter-Optimierung (GridSearch)...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)


pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", rf),
])

# Parameter-Grid definieren
param_grid = {
    "model__n_estimators": [100, 300],         # Anzahl der Bäume
    "model__max_depth": [10, 20, None],        # Maximale Tiefe (None = unbegrenzt)
    "model__min_samples_leaf": [2, 4],         # Mindestanzahl Samples pro Blatt (gegen Overfitting)
}

# GridSearch mit 5-fold Cross-Validation
hazard_precision_scorer = make_scorer(precision_score, pos_label="hazard")

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring=hazard_precision_scorer,
    n_jobs=-1,
    verbose=2
)

# Training auf den 50% Trainingsdaten
grid_search.fit(X_train, y_train)

print("\nBeste Parameter gefunden:")
print(grid_search.best_params_)
print(f"Bester CV-Score (F1-Weighted): {grid_search.best_score_:.4f}")

# Bestes Modell übernehmen
clf = grid_search.best_estimator_

# Feature Importance Analyse


# Zugriff auf das trainierte RF-Modell innerhalb der Pipeline
best_rf = clf.named_steps["model"]
importances = best_rf.feature_importances_

# Feature Namen holen (Reihenfolge ist identisch mit numeric_features, da wir nur num. haben)
feature_names = numeric_features

# DataFrame für schöne Ausgabe
df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
df_imp = df_imp.sort_values(by="Importance", ascending=False)

print("\n=== Feature Importance (Top 10) ===")
print(df_imp.head(10))

# Optional: Plot speichern
try:
    import matplotlib.pyplot as plt
    # import seaborn as sns # Avoid dependency if not present
    
    plt.figure(figsize=(10, 6))
    plt.barh(df_imp.head(10)["Feature"], df_imp.head(10)["Importance"], color='skyblue')
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.gca().invert_yaxis() # Top feature oben
    plt.tight_layout()
    plt.savefig("/Users/kiliandorn/Desktop/Universität/python/DeepAlpine/feature_importance_beta.png")
    print("\nFeature Importance Plot gespeichert: feature_importance.png")
except Exception as e:
    print(f"\nKonnte Plot nicht erstellen (Fehler: {e})")

# Evaluation (Validation Data)


print("\n--- Evaluation auf den Validation-Daten (50%) ---")
y_pred = clf.predict(X_val_for_later)
print(classification_report(y_val_for_later, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val_for_later, y_pred))


# 8. Modell speichern

model_path = "/Users/kiliandorn/Desktop/Universität/python/DeepAlpine/rf_landslides_beta.joblib"
joblib.dump(clf, model_path)
print(f"\nOptimiertes Modell gespeichert unter: {model_path}")
