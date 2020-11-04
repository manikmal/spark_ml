from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, Normalizer, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Bucketizer
from pyspark.ml.stat import Correlation
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
import random

# spark context
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()


# read our clean csv
df_filtered = spark.read.csv('clean_df.csv')

# vector assembler
vectorAssembler = VectorAssembler(inputCols=["HOURLYWindSpeed","","HOURLYStationPressure"],
                                  outputCol="features")
df_pipeline = vectorAssembler.transform(df_filtered)

# checking correlations
Correlation.corr(df_pipeline,"features").head()[0].toArray()

# train test split 
splits = df_filtered.randomSplit([0.8, 0.2])
df_train = splits[0]
df_test = splits[1]

# discretize the value using the Bucketizer, where we split the column in buckets from above 0, 180 and then infinity
bucketizer = Bucketizer(splits=[ 0, 180, float('Inf') ],inputCol="HOURLYWindDirection", outputCol="HOURLYWindDirectionBucketized")
# after the bucketizer we do one hot enncoding 
encoder = OneHotEncoder(inputCol="HOURLYWindDirectionBucketized", outputCol="HOURLYWindDirectionOHE")

# funtion for ccuracy calculation
def classification_metrics(prediction):
    mcEval = MulticlassClassificationEvaluator().setMetricName("accuracy") .setPredictionCol("prediction").setLabelCol("HOURLYWindDirectionBucketized")
    accuracy = mcEval.evaluate(prediction)
    print("Accuracy on test data = %g" % accuracy)

# logistic regression
# defining the model
lr = LogisticRegression(labelCol="HOURLYWindDirectionBucketized", maxIter=10)

# new vector assembler
vectorAssembler = VectorAssembler(inputCols=["HOURLYWindSpeed","HOURLYDRYBULBTEMPC"],
                                  outputCol="features")

# bew piplineline for lr
pipeline = Pipeline(stages=[bucketizer,vectorAssembler,normalizer,lr])

# predictions
model = pipeline.fit(df_train)
prediction = model.transform(df_test)
classification_metrics(prediction)

