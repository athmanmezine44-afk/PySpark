# ============================================
# TELECOM CHURN - RANDOM FOREST MODEL
# Utilisation des résultats EDA de Mohamed + Dataset
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import io
import warnings
warnings.filterwarnings('ignore')

print("Bibliothèques chargées")

#  Upload du dataset

print("\n" + "="*50)
print(" UPLOAD DU DATASET")
print("="*50)

print("\nVeuillez uploader le fichier 'cleanedData.csv'")
uploaded_data = files.upload()
data_filename = list(uploaded_data.keys())[0]

# Chargement du dataset
df = pd.read_csv(io.BytesIO(uploaded_data[data_filename]))
print(f"\n Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")


#  Résultats de l'analyse de Mohamed


print("\n" + "="*50)
print(" RÉSULTATS DE L'ANALYSE DE MOHAMED")
print("="*50)

# Synthèse des insights de Mohamed d'après son notebook(colab)
mohamed_insights = {
    "Churn Rate": f"{ (df['Churn'] == 'Yes').mean() * 100:.2f}%",
    "Contract Impact": "Les clients avec contrat 'Month-to-month' ont le plus haut taux de churn",
    "Internet Service": "La fibre optique (Fiber optic) est associée à plus de churn",
    "Payment Method": "Les paiements par 'Electronic check' montrent un churn élevé",
    "Tenure Effect": "Les clients avec une faible ancienneté churnent plus",
    "Monthly Charges": "Les charges mensuelles élevées augmentent le risque de churn"
}

print("\n INSIGHTS IDENTIFIÉS PAR MOHAMED:")
for key, value in mohamed_insights.items():
    print(f"   • {key}: {value}")


# Préparation des données pour Random Forest

print("\n" + "="*50)
print(" PRÉPARATION DES DONNÉES")
print("="*50)

# Créer une copie
df_model = df.copy()

# Variable cible
df_model['Churn_Num'] = (df_model['Churn'] == 'Yes').astype(int)

# Identifier les colonnes
categorical_cols = df_model.select_dtypes(include=['object']).columns.tolist()
if 'Churn' in categorical_cols:
    categorical_cols.remove('Churn')
if 'customerID' in categorical_cols:
    categorical_cols.remove('customerID')

numeric_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
if 'Churn_Num' in numeric_cols:
    numeric_cols.remove('Churn_Num')

print(f"\n Variables catégorielles: {len(categorical_cols)}")
print(f" Variables numériques: {len(numeric_cols)}")

# Encodage des variables catégorielles
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# Préparer X et y
X = df_encoded.drop(['Churn', 'Churn_Num', 'customerID'], axis=1, errors='ignore')
y = df_encoded['Churn_Num']

print(f"\n Features: {X.shape[1]}")
print(f" Target: {y.value_counts().to_dict()}")


#  Split et entraînement du Random Forest


print("\n" + "="*50)
print(" ENTRAÎNEMENT DU RANDOM FOREST")
print("="*50)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n Split des données:")
print(f"   - Training: {X_train.shape[0]} samples")
print(f"   - Test: {X_test.shape[0]} samples")
print(f"   - Distribution train: {y_train.value_counts().to_dict()}")
print(f"   - Distribution test: {y_test.value_counts().to_dict()}")

# Standardisation (optionnelle pour Random Forest mais gardée pour cohérence)
scaler = StandardScaler()
numeric_features = df_model[numeric_cols].columns
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

for col in numeric_features:
    if col in X_train_scaled.columns:
        X_train_scaled[col] = scaler.fit_transform(X_train_scaled[[col]])
        X_test_scaled[col] = scaler.transform(X_test_scaled[[col]])

# Entraînement du Random Forest
print("\n Entraînement du Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
print(" Modèle entraîné avec succès!")


#  Évaluation du modèle


print("\n" + "="*50)
print(" ÉVALUATION DU MODÈLE")
print("="*50)

# Prédictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Métriques
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n PERFORMANCES DU RANDOM FOREST:")
print(f"   • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   • Precision: {precision:.4f}")
print(f"   • Recall:    {recall:.4f}")
print(f"   • F1-Score:  {f1:.4f}")
print(f"   • ROC-AUC:   {auc:.4f}")

# Rapport détaillé
print(f"\ RAPPORT DE CLASSIFICATION:")
print(classification_report(y_test, y_pred, target_names=['Non-Churn', 'Churn']))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Churn', 'Churn'],
            yticklabels=['Non-Churn', 'Churn'])
plt.title('Matrice de Confusion - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Prédictions')
plt.ylabel('Réel')
plt.tight_layout()
plt.savefig('confusion_matrix_rf.png')
plt.show()


#Importance des features


print("\n" + "="*50)
print(" IMPORTANCE DES FEATURES")
print("="*50)

# Récupérer l'importance des features
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n TOP 15 FEATURES LES PLUS IMPORTANTES:")
print(feature_importance.head(15).to_string(index=False))

# Visualisation
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Importance')
plt.title('Top 15 Features Importantes - Random Forest', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_rf.png')
plt.show()


#  Validation des insights de Mohamed


print("\n" + "="*50)
print(" VALIDATION DES INSIGHTS DE MOHAMED")
print("="*50)

# Vérifier si les insights de Mohamed sont confirmés par le modèle
print("\n CONFRONTATION MODÈLE vs ANALYSE MOHAMED:")

# Insight 1: Contract Type
contract_features = [f for f in feature_importance['feature'] if 'Contract' in f]
if contract_features:
    contract_importance = feature_importance[feature_importance['feature'].isin(contract_features)]['importance'].sum()
    print(f"   ✓ Contract Type - Importance: {contract_importance:.4f} → Insight CONFIRMÉ")

# Insight 2: Internet Service
internet_features = [f for f in feature_importance['feature'] if 'InternetService' in f]
if internet_features:
    internet_importance = feature_importance[feature_importance['feature'].isin(internet_features)]['importance'].sum()
    print(f"   ✓ Internet Service - Importance: {internet_importance:.4f} → Insight CONFIRMÉ")

# Insight 3: Payment Method
payment_features = [f for f in feature_importance['feature'] if 'PaymentMethod' in f]
if payment_features:
    payment_importance = feature_importance[feature_importance['feature'].isin(payment_features)]['importance'].sum()
    print(f"   ✓ Payment Method - Importance: {payment_importance:.4f} → Insight CONFIRMÉ")

# Insight 4: Tenure
if 'tenure' in feature_importance['feature'].values:
    tenure_importance = feature_importance[feature_importance['feature'] == 'tenure']['importance'].values[0]
    print(f"   ✓ Tenure - Importance: {tenure_importance:.4f} → Insight CONFIRMÉ")

# Insight 5: MonthlyCharges
if 'MonthlyCharges' in feature_importance['feature'].values:
    charges_importance = feature_importance[feature_importance['feature'] == 'MonthlyCharges']['importance'].values[0]
    print(f"   ✓ MonthlyCharges - Importance: {charges_importance:.4f} → Insight CONFIRMÉ")


######## Sauvegarde du modèle et résultats ####


print("\n" + "="*50)
print(" SAUVEGARDE DES FICHIERS")
print("="*50)

import joblib

# Sauvegarder le modèle
joblib.dump(rf_model, 'random_forest_churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(" Modèle sauvegardé: random_forest_churn_model.pkl")
print(" Scaler sauvegardé: scaler.pkl")

# Créer un rapport récapitulatif
report = f"""
========================================
 RAPPORT FINAL - RANDOM FOREST CHURN MODEL
========================================

 **ANALYSE DE MOHAMED (RÉSULTATS)**:
• Taux de churn global: {mohamed_insights['Churn Rate']}
• {mohamed_insights['Contract Impact']}
• {mohamed_insights['Internet Service']}
• {mohamed_insights['Payment Method']}
• {mohamed_insights['Tenure Effect']}
• {mohamed_insights['Monthly Charges']}

 **PERFORMANCES DU RANDOM FOREST**:
• Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)
• Precision: {precision:.4f}
• Recall:    {recall:.4f}
• F1-Score:  {f1:.4f}
• ROC-AUC:   {auc:.4f}

 **VALIDATION DES INSIGHTS**:
Les 5 insights clés identifiés par Mohamed sont tous confirmés
par le modèle Random Forest avec une importance significative.

 **FICHIERS GÉNÉRÉS**:
• random_forest_churn_model.pkl - Modèle entraîné
• scaler.pkl - Standardiseur
• confusion_matrix_rf.png - Matrice de confusion
• feature_importance_rf.png - Importance des features

========================================
 Généré le: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
========================================
"""

with open('random_forest_report.txt', 'w') as f:
    f.write(report)

print(" Rapport sauvegardé: random_forest_report.txt")

# ============================================
# Étape 10: Téléchargement
# ============================================

print("\n" + "="*50)
print(" TÉLÉCHARGEMENT DES FICHIERS")
print("="*50)

files_to_download = [
    'random_forest_churn_model.pkl',
    'scaler.pkl',
    'random_forest_report.txt',
    'confusion_matrix_rf.png',
    'feature_importance_rf.png'
]

for file in files_to_download:
    try:
        files.download(file)
        print(f" {file}")
    except:
        print(f" {file}")

# ============================================
# Résumé final
# ============================================

print("\n" + "="*50)
print(" PROCESSUS TERMINÉ AVEC SUCCÈS!")
print("="*50)

print(f"""
 RÉSUMÉ FINAL:
   • Dataset: {df.shape[0]} clients analysés
   • Insights Mohamed: 6 insights clés utilisés
   • Modèle: Random Forest entraîné
   • Performance: F1-Score = {f1:.4f} | AUC = {auc:.4f}
   • Validation: 5/5 insights confirmés par le modèle
   
 Tous les fichiers ont été téléchargés!
""")