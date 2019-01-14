from pyspark import SparkContext
from pyspark.sql import SparkSession

spark = SparkSession.builder \
        .master("local") \
        .appName("covertype data") \
        .getOrCreate()
        

covertype_df = spark.read.csv('../covertype2/train.csv',header=True, inferSchema=True).repartition(100)
# only 3 rows in train.csv has soil_type: 'unspecified in the USFS Soil and ELU Survey.', while in test.csv doesn't
# have this soil type, so we delete these 3 rows in train.csv for training
covertype_df = covertype_df.filter(covertype_df.Soil_Type != 'unspecified in the USFS Soil and ELU Survey.')

# transform the two categorical feature
from pyspark.ml.feature import StringIndexer

stringIndexer = StringIndexer(inputCol = "Soil_Type", outputCol = "Soil_Index")
model1 = stringIndexer.fit(covertype_df)
indexedDF = model1.transform(covertype_df)

stringIndexer2 = StringIndexer(inputCol = "Wild_Type", outputCol = "Wild_Index")
model2 = stringIndexer2.fit(indexedDF)
indexedDF2 = model2.transform(indexedDF)

from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(inputCol = "Soil_Index", outputCol = "SoilEncoder")
encoder.setDropLast(False)
encodedDF = encoder.transform(indexedDF2)

encoder2 = OneHotEncoder(inputCol = "Wild_Index", outputCol = "WildEncoder")
encoder2.setDropLast(False)
encodedDF2 = encoder2.transform(encodedDF)

#Use the VectorAssembler technique to accumulate all features into one vector. 
from pyspark.ml.feature import VectorAssembler

vector_assembler = VectorAssembler(inputCols=['SoilEncoder', # feature name of Soil type encoded
                                              'WildEncoder', # feature name of Wild type encoded
                                              'Elevation',
                                              'Aspect',
                                              'Slope',
                                              'Horizontal_Distance_To_Hydrology',
                                              'Vertical_Distance_To_Hydrology',
                                              'Horizontal_Distance_To_Roadways',
                                              'Hillshade_9am',
                                              'Hillshade_Noon',
                                              'Hillshade_3pm',
                                              'Horizontal_Distance_To_Fire_Points'
                                              ], outputCol='features')
finalDF = vector_assembler.transform(encodedDF2)
# unique wild_type : ['Comanche', 'Rawah', 'Cache', 'Neota']

#Fit the Random Forest model to the train dataset.
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol='Target',featuresCol='features', numTrees=100,maxDepth=9)

(trainingData, testData) = finalDF.randomSplit([0.8, 0.2], seed = 123)
model = rf.fit(trainingData)

#model.save('randomforest_model')
#model = RandomForestClassifier.load('randomforest_model')

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
predictions = model.transform(testData)
evaluator = MulticlassClassificationEvaluator(labelCol = "Target", predictionCol = "prediction", metricName = "accuracy")
accuracy = evaluator.evaluate(predictions)

#perform on test data
dfTest = spark.read.csv('../covertype2/test.csv',header=True, inferSchema=True).repartition(100)

stringIndexer = StringIndexer(inputCol = "Soil_Type", outputCol = "Soil_Index")
model1 = stringIndexer.fit(dfTest)
indexedDF = model1.transform(dfTest)

stringIndexer2 = StringIndexer(inputCol = "Wild_Type", outputCol = "Wild_Index")
model2 = stringIndexer2.fit(indexedDF)
indexedDF2 = model2.transform(indexedDF)

encoder = OneHotEncoder(inputCol = "Soil_Index", outputCol = "SoilEncoder")
encoder.setDropLast(False)
encodedDF = encoder.transform(indexedDF2)

encoder2 = OneHotEncoder(inputCol = "Wild_Index", outputCol = "WildEncoder")
encoder2.setDropLast(False)
encodedDF2Test = encoder2.transform(encodedDF)

finalDFTest = vector_assembler.transform(encodedDF2Test)

# Calculate accuracy
predictions_test = model.transform(finalDFTest)
evaluator = MulticlassClassificationEvaluator(labelCol = "Target", predictionCol = "prediction", metricName = "accuracy")
accuracy = evaluator.evaluate(predictions_test)
print(round(accuracy,2))
