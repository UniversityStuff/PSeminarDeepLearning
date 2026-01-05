import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

train_file = 'landslides_training_data.csv'
pred_126_file = 'PREDICT_SSP126_READY.csv'
pred_585_file = 'PREDICT_SSP585_READY.csv'

features = [
    'Elevation', 'Slope', 'Aspect',
    'BIO01_Historical_Mean', 'BIO05_Historical_Max', 'BIO06_Historical_Min',
    'BIO12_Historical_Prec', 'BIO13_Historical_Prec', 'BIO15_Historical_Prec'
]

df = pd.read_csv(train_file)

for f in features:
    if f not in df.columns:
        df[f] = 0

X = df[features].fillna(0)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
rec_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='recall')

print(f"CV Accuracy: {acc_scores.mean()*100:.1f}%")
print(f"CV Recall:   {rec_scores.mean()*100:.1f}%")

rf.fit(X_train, y_train)

probs_test = rf.predict_proba(X_test)[:, 1]
y_pred = (probs_test >= 0.4).astype(int)

print("\nTEST REPORT:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Danger']))

imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=imp.values, y=imp.index, palette='viridis')
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

def predict_scenario(in_path, out_path):
    if not os.path.exists(in_path):
        print(f"File not found: {in_path}")
        return None
    
    df_fut = pd.read_csv(in_path)
    
    for f in features:
        if f not in df_fut.columns:
            df_fut[f] = 0
            
    probs = rf.predict_proba(df_fut[features].fillna(0))[:, 1]
    df_fut['Landslide_Probability'] = probs
    
    df_fut['Risk_Class'] = df_fut['Landslide_Probability'].apply(
        lambda x: 'HIGH' if x >= 0.6 else ('MEDIUM' if x >= 0.3 else 'LOW')
    )
    
    df_fut.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df_fut

df_126 = predict_scenario(pred_126_file, 'RESULT_SSP126.csv')
df_585 = predict_scenario(pred_585_file, 'RESULT_SSP585.csv')

if df_126 is not None and df_585 is not None:
    diff = df_585['Landslide_Probability'] - df_126['Landslide_Probability']
    
    print(f"\nWorse (>10%): {sum(diff > 0.1)}")
    print(f"Better (>10%):  {sum(diff < -0.1)}")

    plt.figure(figsize=(10, 8))
    plt.scatter(df_126['Longitude'], df_126['Latitude'], c=diff, cmap='seismic', vmin=-0.3, vmax=0.3, s=15, alpha=0.8)
    plt.colorbar(label="Difference (Red = Higher Risk in SSP585)")
    plt.title("Difference SSP585 - SSP126")
    plt.savefig('difference_map.png')
    plt.show()

print("Done.")
