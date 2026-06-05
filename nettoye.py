from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when, count
from pyspark.sql.types import DoubleType

# =====================================================
# 1. Création de la session Spark
# =====================================================

spark = SparkSession.builder \
    .appName("Telecom Customer Churn - Data Engineering") \
    .getOrCreate()

# =====================================================
# 2. Chargement du dataset
# =====================================================

df = spark.read.csv(
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    header=True,
    inferSchema=True
)

print("=" * 50)
print("DATASET ORIGINAL")
print("=" * 50)

print(f"Nombre de lignes : {df.count()}")
print(f"Nombre de colonnes : {len(df.columns)}")

df.printSchema()

# =====================================================
# 3. Vérification des valeurs vides dans TotalCharges
# =====================================================

empty_totalcharges = df.filter(
    trim(col("TotalCharges")) == ""
).count()

print(f"\nValeurs vides dans TotalCharges : {empty_totalcharges}")

# =====================================================
# 4. Remplacement des espaces vides par NULL
# =====================================================

df = df.withColumn(
    "TotalCharges",
    when(
        trim(col("TotalCharges")) == "",
        None
    ).otherwise(col("TotalCharges"))
)

# =====================================================
# 5. Conversion de TotalCharges en Double
# =====================================================

df = df.withColumn(
    "TotalCharges",
    col("TotalCharges").cast(DoubleType())
)

# =====================================================
# 6. Comptage des valeurs NULL
# =====================================================

print("\nValeurs manquantes par colonne :")

df.select([
    count(
        when(col(c).isNull(), c)
    ).alias(c)
    for c in df.columns
]).show(truncate=False)

# =====================================================
# 7. Suppression des lignes contenant des NULL
# =====================================================

before_null = df.count()

df = df.na.drop()

after_null = df.count()

print(f"Lignes avant suppression des NULL : {before_null}")
print(f"Lignes après suppression des NULL : {after_null}")

# =====================================================
# 8. Suppression des doublons
# =====================================================

before_duplicates = df.count()

df = df.dropDuplicates()

after_duplicates = df.count()

print(f"\nLignes avant suppression des doublons : {before_duplicates}")
print(f"Lignes après suppression des doublons : {after_duplicates}")

# =====================================================
# 9. Vérification de l'unicité des CustomerID
# =====================================================

duplicate_ids = df.groupBy("customerID") \
                  .count() \
                  .filter(col("count") > 1)

print("\nCustomerID dupliqués :")
duplicate_ids.show()

# =====================================================
# 10. Contrôle qualité final
# =====================================================

print("\n" + "=" * 50)
print("DATASET NETTOYÉ")
print("=" * 50)

print(f"Nombre final de lignes : {df.count()}")
print(f"Nombre final de colonnes : {len(df.columns)}")

df.printSchema()

print("\nStatistiques descriptives :")
df.describe().show()

# =====================================================
# 11. Sauvegarde du dataset nettoyé
# =====================================================

df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("clean_telco_data_csv")

df.write \
    .mode("overwrite") \
    .parquet("clean_telco_data_parquet")

print("\nDataset nettoyé sauvegardé avec succès !")

# =====================================================
# 12. Fermeture de Spark
# =====================================================

spark.stop()