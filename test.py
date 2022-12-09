import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import findspark
import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
import warnings
warnings.filterwarnings("ignore")

findspark.init()
findspark.find()

conf = pyspark.SparkConf().setAppName('winequality').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

df = spark.read.format("csv").load("ValidationDataset.csv", header=True, sep=";")
df.printSchema()
df.show(5)

for col_name in df.columns[1:-1] + ['""""quality"""""']:
    df = df.withColumn(col_name, col(col_name).cast('float'))
df = df.withColumnRenamed('""""quality"""""', "quality")

features = np.array(df.select(df.columns[1:-1]).collect())
label = np.array(df.select('quality').collect())

# Function to create label points
def to_labeled_point(features, labels):
    lp = []
    for x, y in zip(features, labels):
        labeled_points = LabeledPoint(y, x)
        lp.append(labeled_points)
    return lp

# Function to convert to RDD
def to_rdd(sc, labeled_points):
    return sc.parallelize(labeled_points)

data_lp = to_labeled_point(features, label)

data_lp_rdd = to_rdd(sc, data_lp)

RF = RandomForestModel.load(sc, "/home/hadoop/wine_quality/")

print("model successfully loaded")

pred = RF.predict(data_lp_rdd.map(lambda x: x.features))

pred_rdd = data_lp_rdd.map(lambda y: y.label).zip(pred)
pred_df = pred_rdd.toDF()

quality_pred = pred_rdd.toDF(["quality", "prediction"])
quality_pred.show(5)
quality_pred_df = quality_pred.toPandas()


print("---------------Performance Metrics-----------------")

print("Accuracy : ", accuracy_score(quality_pred_df['quality'], quality_pred_df['prediction']))

print("F1- score : ", f1_score(quality_pred_df['quality'], quality_pred_df['prediction'], average='weighted'))

print("Confusion Matrix : ", confusion_matrix(quality_pred_df['quality'], quality_pred_df['prediction']))

print("Classification Report : ", classification_report(quality_pred_df['quality'], quality_pred_df['prediction']))

test_error = pred_rdd.filter(
    lambda y: y[0] != y[1]).count() / float(data_lp_rdd.count())
print('Test Error : ' + str(test_error))
