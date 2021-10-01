# importing the os package to add a package to pyspark submit
import os
# this will add the databrick's sparkdl package to the pyspark submit function.
# This package will allow us to use the sparkdl package, and the functions like DeepImageFeaturizer
# to perform deep learning and image classification
SUBMIT_ARGS = "--packages databricks:spark-deep-learning:1.0.0-spark2.3-s_2.11 pyspark-shell"
# This will add that package to the arguments used when the spark-submit is called
os.environ["PYSPARK_SUBMIT_ARGS"] = SUBMIT_ARGS
# Setting up the spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("DL image recognition").getOrCreate()
sc = spark.sparkContext
# importing the required pyspark librarues
from pyspark.ml.image import ImageSchema
from pyspark.sql import Row
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import monotonically_increasing_id
# path to the training images in AWS S3 bucket
train15Img ="s3://final-project-657/resized-train-15/"
# path to the training labels in AWS S3 bucket
labelDf = spark.read.csv("s3://final-project-657/trainLabels15.csv")
# This will go through the labels DF, and give column names to each row
labels = labelDf.rdd.map(lambda r: Row(imgName=str(r._c0), label=int(r._c1)))
# add an id to each corresponding label in the labels data frame
labelsDf = spark.createDataFrame(labels).withColumn("id", monotonically_increasing_id())
labelsDf = labelsDf.drop("imgName")
print(labelsDf.show())
# This will convert the label in the dataframe to an Integer type from a string type
changedTypedf = labelsDf.withColumn("label", labelsDf["label"].cast(IntegerType()))
# This will use the ImageSchema library to read in the images.
# This readImages function will read in each image as an image struct, and we set an column with a unique id with each image
train15Eyes = ImageSchema.readImages(train15Img).withColumn("id", monotonically_increasing_id())
print(train15Eyes.show())
# This will join the labels df and the image df on that id, so each label is correctly corresponded with the images
dataDf = train15Eyes.join(changedTypedf, on="id")
# this will split the data into a 70% train data, and 30% test data
trainDf, testDf = dataDf.randomSplit([0.7, 0.3])
print(trainDf.show())
print("Initializing featurizers and LR nd pipeline")
# This will import the required libraries to perform the image recognition
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer 
# Using the deep image featureizer to convert the images into numerical vectors
# DeepImageFeaturizer is from the databricks package
# It will fit each image into the InceptionV3 convolutional Neural Network, and convert them into numerical vectors
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
# This will inialize the logistic regression model with the best parameters from the cross validation results
lr = LogisticRegression(maxIter=10, regParam=0.2, elasticNetParam=0.7, labelCol="label")
# The pipeline is initialized with both stages passed through it (which are the image featurizer and Logistic regression)
p = Pipeline(stages=[featurizer, lr])
print("fitting")
# This will fit the training data into the pipeline
p_model = p.fit(trainDf)
# This will predict the values of test dataframe using the pipline
predDf = p_model.transform(testDf)
# selecting only the prediction and the label from the resulting dataframe
predAndLabels = tested_df.select("prediction", "label")
pl = predAndLabels.rdd.map(lambda x: (x.prediction, x.label))
# Saving the tuples to a text file for further analysis of results
pl.coalesce(1).saveAsTextFile("s3://final-project-657/Predictions/results")
# This will open the results file with contains all the tuples of prediction and labels
results = open("s3://final-project-657/Predictions/results/part-00000")
# initializng the variables with total count and the correct number of counts
totalCount = 0
correctCount = 0
# This will go through each tuple in the results file
for line in results:
    # converts each line into a tuple with integer and double
    tup = eval(line)
    # Checks to see if the prediction and the label match
    if int(tup[0]) == int(tup[1]):
        correctCount += 1
    totalCount += 1
# This will get the accuracy based on the number of correct counts over the total number of predictions
accuracy = correctCount/totalCount
print("----- Results -----")
print("Accuracy = " + str(accuracy))











