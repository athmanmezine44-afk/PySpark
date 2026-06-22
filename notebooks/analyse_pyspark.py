

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    avg,
    when,
    desc,
    round as spark_round,
    sum as spark_sum,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os

          #DÉMARRER LA SESSION SPARK
print("\n" + "=" * 60)
print("  ÉTAPE 1 — Démarrage de la session Spark")
print("=" * 60)

spark = SparkSession.builder \
    .appName("TelcoChurnAnalysis") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(f"✅ Spark démarré — version : {spark.version}")

              #CHARGER LE DATASET

print("\n" + "=" * 60)
print("  ÉTAPE 2 — Chargement du dataset")
print("=" * 60)

# Chemin relatif depuis la racine du projet
DATA_PATH  = "data/cleanedData.csv"
OUTPUT_DIR = "data"

df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

nb_lignes   = df.count()
nb_colonnes = len(df.columns)
print(f"✅ Dataset chargé : {nb_lignes} lignes, {nb_colonnes} colonnes")
print("\n── Schéma du dataset ──")
df.printSchema()

                        #EXPLORATION INITIALE
print("\n" + "=" * 60)
print("  ÉTAPE 3 — Exploration initiale")
print("=" * 60)

print("\n── Aperçu des 5 premières lignes ──")
df.show(5, truncate=False)

print("\n── Statistiques descriptives (colonnes numériques) ──")
df.describe(["tenure", "MonthlyCharges", "TotalCharges"]).show()

print("\n── Vérification des valeurs nulles ──")
null_counts = df.select([
    spark_sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
])
null_counts.show()

           #DISTRIBUTION DU CHURN GLOBAL

print("\n" + "=" * 60)
print("  ÉTAPE 4 — Distribution du Churn Global")
print("=" * 60)

total = df.count()

churn_global = df.groupBy("Churn").agg(
    count("*").alias("Nombre_Clients")
).withColumn(
    "Pourcentage_%", spark_round(col("Nombre_Clients") / total * 100, 2)
).orderBy("Churn")

print("\n── Résultat ──")
churn_global.show()

# Export CSV pour Streamlit
df_global_pd = churn_global.toPandas()
df_global_pd.to_csv(f"{OUTPUT_DIR}/analyse_churn_global.csv", index=False)
print("📁 Exporté : analyse_churn_global.csv")

                            #CHURN PAR TYPE DE CONTRAT

print("\n" + "=" * 60)
print("  ÉTAPE 5 — Churn par Type de Contrat")
print("=" * 60)

churn_contrat = df.groupBy("Contract").agg(
    count("*").alias("Total"),
    spark_sum(when(col("Churn") == "Yes", 1).otherwise(0)).alias("Churned"),
    spark_sum(when(col("Churn") == "No",  1).otherwise(0)).alias("Active"),
).withColumn(
    "Taux_Churn_%", spark_round(col("Churned") / col("Total") * 100, 1)
).orderBy(desc("Taux_Churn_%"))

print("\n── Résultat ──")
churn_contrat.show()

df_contrat_pd = churn_contrat.toPandas()
df_contrat_pd.to_csv(f"{OUTPUT_DIR}/analyse_churn_contrat.csv", index=False)
print("📁 Exporté : analyse_churn_contrat.csv")

                                  #CHURN PAR SERVICE INTERNET

print("\n" + "=" * 60)
print("  ÉTAPE 6 — Churn par Service Internet")
print("=" * 60)

churn_internet = df.groupBy("InternetService").agg(
    count("*").alias("Total"),
    spark_sum(when(col("Churn") == "Yes", 1).otherwise(0)).alias("Churned"),
    spark_sum(when(col("Churn") == "No",  1).otherwise(0)).alias("Active"),
).withColumn(
    "Taux_Churn_%", spark_round(col("Churned") / col("Total") * 100, 1)
).orderBy(desc("Taux_Churn_%"))

print("\n── Résultat ──")
churn_internet.show()

df_internet_pd = churn_internet.toPandas()
df_internet_pd.to_csv(f"{OUTPUT_DIR}/analyse_churn_internet.csv", index=False)
print("📁 Exporté : analyse_churn_internet.csv")

                             #CHURN PAR MODE DE PAIEMENT

print("\n" + "=" * 60)
print("  ÉTAPE 7 — Churn par Mode de Paiement")
print("=" * 60)

churn_paiement = df.groupBy("PaymentMethod").agg(
    count("*").alias("Total"),
    spark_sum(when(col("Churn") == "Yes", 1).otherwise(0)).alias("Churned"),
    spark_sum(when(col("Churn") == "No",  1).otherwise(0)).alias("Active"),
).withColumn(
    "Taux_Churn_%", spark_round(col("Churned") / col("Total") * 100, 1)
).orderBy(desc("Taux_Churn_%"))

print("\n── Résultat ──")
churn_paiement.show(truncate=False)

df_paiement_pd = churn_paiement.toPandas()
df_paiement_pd.to_csv(f"{OUTPUT_DIR}/analyse_churn_paiement.csv", index=False)
print("📁 Exporté : analyse_churn_paiement.csv")

                        #CHARGES MOYENNES : CHURNED VS ACTIFS

print("\n" + "=" * 60)
print("  ÉTAPE 8 — Charges moyennes : Churned vs Actifs")
print("=" * 60)

charges_churn = df.groupBy("Churn").agg(
    spark_round(avg("MonthlyCharges"), 2).alias("Avg_MonthlyCharges"),
    spark_round(avg("TotalCharges"),   2).alias("Avg_TotalCharges"),
    spark_round(avg("tenure"),         1).alias("Avg_Tenure_Mois"),
).orderBy("Churn")

print("\n── Résultat ──")
charges_churn.show()

df_charges_pd = charges_churn.toPandas()
df_charges_pd.to_csv(f"{OUTPUT_DIR}/analyse_charges_churn.csv", index=False)
print("📁 Exporté : analyse_charges_churn.csv")

                            #CHURN PAR PROFIL CLIENT

print("\n" + "=" * 60)
print("  ÉTAPE 9 — Churn par Profil Client")
print("=" * 60)

profils_config = {
    "SeniorCitizen": {0: "Non-Senior", 1: "Senior"},
    "Partner":       {"No": "Sans partenaire", "Yes": "Avec partenaire"},
    "Dependents":    {"No": "Sans dépendants", "Yes": "Avec dépendants"},
}

resultats_profils = []
for variable, mapping in profils_config.items():
    res = df.groupBy(variable).agg(
        count("*").alias("Total"),
        spark_sum(when(col("Churn") == "Yes", 1).otherwise(0)).alias("Churned"),
    ).withColumn(
        "Taux_Churn_%", spark_round(col("Churned") / col("Total") * 100, 1)
    )
    print(f"\n── {variable} ──")
    res.show()
    tmp = res.toPandas()
    tmp["Categorie"] = tmp[variable].map(
        {str(k): v for k, v in mapping.items()}
    ).fillna(tmp[variable].astype(str))
    tmp["Variable"] = variable
    resultats_profils.append(tmp[["Variable", "Categorie", "Total", "Churned", "Taux_Churn_%"]])

churn_profils_pd = pd.concat(resultats_profils, ignore_index=True)
churn_profils_pd.to_csv(f"{OUTPUT_DIR}/analyse_churn_profils.csv", index=False)
print("\n📁 Exporté : analyse_churn_profils.csv")

    
                            #CHURN PAR SERVICES ADDITIONNELS

print("\n" + "=" * 60)
print("  ÉTAPE 10 — Churn par Services Additionnels")
print("=" * 60)

services = [
    "OnlineSecurity",
    "TechSupport",
    "OnlineBackup",
    "DeviceProtection",
    "StreamingTV",
    "StreamingMovies",
]

rows_services = []
for service in services:
    # On filtre les clients qui ont internet (exclut "No internet service")
    df_filtre = df.filter(col(service) != "No internet service")
    res = df_filtre.groupBy(service).agg(
        count("*").alias("Total"),
        spark_sum(when(col("Churn") == "Yes", 1).otherwise(0)).alias("Churned"),
    ).withColumn(
        "Taux_Churn_%", spark_round(col("Churned") / col("Total") * 100, 1)
    )
    # Taux chez les clients N'ayant PAS souscrit au service
    row_no = res.filter(col(service) == "No").collect()
    # Taux chez les clients AYANT souscrit au service
    row_yes = res.filter(col(service) == "Yes").collect()
    if row_no and row_yes:
        rows_services.append({
            "Service":            service,
            "Sans_service_%":     row_no[0]["Taux_Churn_%"],
            "Avec_service_%":     row_yes[0]["Taux_Churn_%"],
        })

df_services_pd = pd.DataFrame(rows_services) \
    .sort_values("Sans_service_%", ascending=False)

print("\n── Résultat ──")
print(df_services_pd.to_string(index=False))

df_services_pd.to_csv(f"{OUTPUT_DIR}/analyse_churn_services.csv", index=False)
print("\n📁 Exporté : analyse_churn_services.csv")


                            #VISUALISATIONS MATPLOTLIB

print("\n" + "=" * 60)
print("  ÉTAPE 11 — Génération des visualisations")
print("=" * 60)

VERT  = "#059669"
ROUGE = "#ef4444"
AMBER = "#f59e0b"

fig = plt.figure(figsize=(18, 20))
fig.suptitle("Analyse Churn — Telco Customer Churn", fontsize=16,
             fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Graphique 1 : Churn global (camembert) ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
labels = df_global_pd["Churn"].map({"No": "Actifs", "Yes": "Churned"})
colors = [VERT if c == "No" else ROUGE for c in df_global_pd["Churn"]]
wedges, texts, autotexts = ax1.pie(
    df_global_pd["Nombre_Clients"],
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("bold")
ax1.set_title("Churn Global", fontweight="bold", fontsize=12, pad=15)

# ── Graphique 2 : Churn par contrat 
ax2 = fig.add_subplot(gs[0, 1])
df_c = df_contrat_pd.sort_values("Taux_Churn_%", ascending=True)
bar_colors = [ROUGE if v > 40 else AMBER if v > 20 else VERT
              for v in df_c["Taux_Churn_%"]]
bars = ax2.barh(df_c["Contract"], df_c["Taux_Churn_%"],
                color=bar_colors, edgecolor="none", height=0.5)
for bar, val in zip(bars, df_c["Taux_Churn_%"]):
    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
             f"{val}%", va="center", fontsize=10, fontweight="bold")
ax2.set_title("Taux de Churn par Contrat", fontweight="bold", fontsize=12)
ax2.set_xlabel("Taux de Churn (%)")
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_xlim(0, df_c["Taux_Churn_%"].max() + 12)

# ── Graphique 3 : Churn par service internet ─────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
df_i = df_internet_pd.sort_values("Taux_Churn_%", ascending=False)
bar_colors_i = [ROUGE if v > 40 else AMBER if v > 10 else VERT
                for v in df_i["Taux_Churn_%"]]
bars_i = ax3.bar(df_i["InternetService"], df_i["Taux_Churn_%"],
                 color=bar_colors_i, edgecolor="none", width=0.5)
for bar, val in zip(bars_i, df_i["Taux_Churn_%"]):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
             f"{val}%", ha="center", fontsize=10, fontweight="bold")
ax3.set_title("Taux de Churn par Service Internet", fontweight="bold", fontsize=12)
ax3.set_ylabel("Taux de Churn (%)")
ax3.spines[["top", "right"]].set_visible(False)
ax3.set_ylim(0, df_i["Taux_Churn_%"].max() + 12)

# ── Graphique 4 : Charges moyennes (Actifs vs Churned) ───────────────────────
ax4 = fig.add_subplot(gs[1, 1])
df_ch = df_charges_pd.copy()
x      = [0, 1]
width  = 0.35
labels_x = ["Charges\nmensuelles ($)", "Ancienneté\n(mois)"]
vals_no  = [
    float(df_ch.loc[df_ch["Churn"] == "No", "Avg_MonthlyCharges"].values[0]),
    float(df_ch.loc[df_ch["Churn"] == "No", "Avg_Tenure_Mois"].values[0]),
]
vals_yes = [
    float(df_ch.loc[df_ch["Churn"] == "Yes", "Avg_MonthlyCharges"].values[0]),
    float(df_ch.loc[df_ch["Churn"] == "Yes", "Avg_Tenure_Mois"].values[0]),
]
b1 = ax4.bar([i - width/2 for i in x], vals_no,  width, label="Actifs",  color=VERT,  edgecolor="none")
b2 = ax4.bar([i + width/2 for i in x], vals_yes, width, label="Churned", color=ROUGE, edgecolor="none")
for bar, val in list(zip(b1, vals_no)) + list(zip(b2, vals_yes)):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")
ax4.set_xticks(x)
ax4.set_xticklabels(labels_x)
ax4.set_title("Actifs vs Churned — Profil Financier", fontweight="bold", fontsize=12)
ax4.legend()
ax4.spines[["top", "right"]].set_visible(False)

# ── Graphique 5 : Services additionnels ──────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :])
df_sv = df_services_pd.sort_values("Sans_service_%", ascending=True)
x_sv    = range(len(df_sv))
width_s = 0.35
b_sans = ax5.bar([i - width_s/2 for i in x_sv], df_sv["Sans_service_%"],
                 width_s, label="Sans le service", color=ROUGE, edgecolor="none")
b_avec = ax5.bar([i + width_s/2 for i in x_sv], df_sv["Avec_service_%"],
                 width_s, label="Avec le service", color=VERT,  edgecolor="none")
for bar in list(b_sans) + list(b_avec):
    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{bar.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
ax5.set_xticks(list(x_sv))
ax5.set_xticklabels(df_sv["Service"], rotation=20, ha="right")
ax5.set_title("Churn selon les Services Additionnels (avec vs sans)",
              fontweight="bold", fontsize=12)
ax5.set_ylabel("Taux de Churn (%)")
ax5.legend()
ax5.spines[["top", "right"]].set_visible(False)

plt.savefig(f"{OUTPUT_DIR}/visualisations_churn.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("📁 Exporté : visualisations_churn.png")
plt.show()


#RÉSUMÉ DES INSIGHTS CLÉS

print("\n" + "=" * 60)
print("  ÉTAPE 12 — Insights Clés")
print("=" * 60)

# Calcul dynamique depuis les données
taux_global    = round(df_global_pd.loc[df_global_pd["Churn"] == "Yes", "Pourcentage_%"].values[0], 1)
contrat_mensuel = df_contrat_pd.loc[df_contrat_pd["Contract"] == "Month-to-month", "Taux_Churn_%"].values[0]
internet_fibre  = df_internet_pd.loc[df_internet_pd["InternetService"] == "Fiber optic", "Taux_Churn_%"].values[0]
service_top_sans = df_services_pd.iloc[0]["Service"]
taux_top_sans    = df_services_pd.iloc[0]["Sans_service_%"]
avg_monthly_churn = df_charges_pd.loc[df_charges_pd["Churn"] == "Yes", "Avg_MonthlyCharges"].values[0]
avg_monthly_actif = df_charges_pd.loc[df_charges_pd["Churn"] == "No",  "Avg_MonthlyCharges"].values[0]
avg_tenure_churn  = df_charges_pd.loc[df_charges_pd["Churn"] == "Yes", "Avg_Tenure_Mois"].values[0]
avg_tenure_actif  = df_charges_pd.loc[df_charges_pd["Churn"] == "No",  "Avg_Tenure_Mois"].values[0]

insights = [
    f"Taux de churn global : {taux_global}% des clients ont quitté l'opérateur",
    f" Contrat mensuel : taux de churn de {contrat_mensuel}% — le plus risqué de tous les contrats",
    f" Fibre optique : {internet_fibre}% de churn — bien plus élevé que DSL",
    f"Sans {service_top_sans} : {taux_top_sans}% de churn — les services additionnels fidélisent",
    f"Clients churned payent en moyenne {avg_monthly_churn}$/mois vs {avg_monthly_actif}$ pour les actifs",
    f"Ancienneté moyenne : {avg_tenure_churn} mois (churned) vs {avg_tenure_actif} mois (actifs) — les nouveaux clients partent plus",
]

print()
for insight in insights:
    print(f"  {insight}")

# Export CSV
pd.DataFrame({"Insight": insights}).to_csv(
    f"{OUTPUT_DIR}/analyse_insights.csv", index=False
)
print("\n📁 Exporté : analyse_insights.csv")


                            #FERMER SPARK

print("\n" + "=" * 60)
print("  ÉTAPE 13 — Fermeture de la session Spark")
print("=" * 60)

spark.stop()
print("✅ Session Spark fermée proprement.")

print("\n" + "=" * 60)
print("  ✅ ANALYSE TERMINÉE — Fichiers exportés dans data/ :")
print("=" * 60)
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.startswith("analyse_") or f == "visualisations_churn.png":
        size_kb = round(os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024, 1)
        print(f"  └─ {f:<40} ({size_kb} Ko)")
print()
print("  👉 Lance maintenant : streamlit run app.py")
print("     La page '🔬 Analyse PySpark' affichera tous ces résultats.")
print("=" * 60 + "\n")
