import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

train_path = r"C:\Users\richt\Desktop\testNaturgefahrenOhneBuffer\prediction_modell\landslides_training_data.csv"

features = [
    'Elevation', 'Slope', 'Aspect',
    'BIO01_Historical_Mean', 'BIO05_Historical_Max', 'BIO06_Historical_Min',
    'BIO12_Historical_Prec', 'BIO13_Historical_Prec', 'BIO15_Historical_Prec'
]

df = pd.read_csv(train_path)

X = df[features].fillna(0)
y = df['Target']

rf = RandomForestClassifier(
    n_estimators=500, 
    max_depth=10, 
    min_samples_leaf=8, 
    random_state=42
)
rf.fit(X, y)

importances = rf.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

print(feature_imp_df)
feature_imp_df.to_csv('Feature_Importance.csv', index=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Feature Importance (Einfluss der Variablen)', fontsize=14)
plt.xlabel('Wichtigkeit (0 bis 1)')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('Feature_Importance_Plot.png', dpi=300)
plt.show()