from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, Normalizer, MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
import pixiedust
from pyspark.sql.functions import col

# pre processing script

# building a spark session
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()

# reading a parquet file 
df = spark.read.parquet('hmp.parquet')

# examining the data set
df.groupBy('class').count().show()

# bar plot using pixie dust
counts = df.groupBy('class').count().orderBy('count')
display(counts)

# pre-processing
# String indexer:
indexer = StringIndexer(inputCol="class", outputCol="classIndex")

# one-hot-encoder:
encoder = OneHotEncoder(inputCol="classIndex", outputCol="categoryVec")

# vector assembler
vectorAssembler = VectorAssembler(inputCols=["x","y","z"],
                                  outputCol="features")

# normalizer:
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)



# pipeline: 
pipeline = Pipeline(stages=[indexer, encoder, vectorAssembler, normalizer])
model = pipeline.fit(df)

# predictions
prediction = model.transform(df)
prediction.show()

