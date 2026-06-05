from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create Spark session
spark = SparkSession.builder \
    .appName("Telco Churn Full Model") \
    .getOrCreate()
# Load data
df = spark.read.csv("D:\spark\PySpark\cleanedData.csv", header=True, inferSchema=True)
df.show(5)
df.printSchema()

#fix types
df = df.withColumn("TotalCharges", col("TotalCharges").cast("double"))
df = df.na.drop()

#label(fixed error)
df = df.withColumn(
    "label",
    when(col("Churn") == "Yes", 1).otherwise(0)
)

#Features selection
categorical_cols = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaymentMethod"
]

numeric_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

#encodage
indexers = [
    StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid="keep")
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(inputCol=c+"_index", outputCol=c+"_vec")
    for c in categorical_cols
]

#feature vector
assembler = VectorAssembler(
    inputCols=[c+"_vec" for c in categorical_cols] + numeric_cols,
    outputCol="features"
)

#model
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100
)

#pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

#split test/train
train, test = df.randomSplit([0.8, 0.2], seed=42)

#train model
model = pipeline.fit(train)

#predictions
predictions = model.transform(test)
predictions.select("features", "label", "prediction").show(5)

#evaluate
evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)
print("AUC =", auc)

#save model
model.save("churn_rf_model")


