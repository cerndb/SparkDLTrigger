//
// Converts the High Level feature classifier training and test datasets to TFRecord format
// - Reads Parquet data into a dataframe
// - Save the dataframe as TFRecords using spark-tensorflow-connector
// (see https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-connector)
// This is used as input for the TensorFlow Keras with TFRecord example notebook
// Run with Scala shell or in a Scala notebook

// Commnad line used to run this (June 2019, updated Aug 2019):

bin/spark-shell --master local[*] --packages org.tensorflow:spark-tensorflow-connector_2.11:1.14.0

// Data source

val PATH = "<...>/SparkDLTRigger/Data/"
val outputPATH = PATH
val df=spark.read.parquet(PATH + "testUndersampled_HLF_features.parquet")

scala> df.printSchema
root
 |-- HLF_input: array (nullable = true)
 |    |-- element: double (containsNull = true)
 |-- encoded_label: array (nullable = true)
 |    |-- element: double (containsNull = true)

// save the test dataset in TFRecord format
// compact output in 2 files with coalesce(2)
df.coalesce(2).write.format("tfrecords").save(outputPATH + "testUndersampled_HLF_features.tfrecord")

//
// Repeat for the training dataset
//

// Read Parquet
val df2=spark.read.parquet(PATH + "trainUndersampled_HLF_features.parquet")

// save the training dataset in TFRecord format
// compact output in 4 files with coalesce(4)
df2.coalesce(4).write.format("tfrecords").save(outputPATH + "trainUndersampled_HLF_features.tfrecord")

