//
// Converts data format for the particle sequence and high level feature classifier datasets.
// Reads from Apache Parquet and writes to TFRecord format
// - Reads Parquet data as a dataframe
// - Convert Spark Vectors to Arrays
// - Save the dataframe as TFRecords using spark-tensorflow-connector
// (see https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-connector)
// This is used as input for the TensorFlow Keras with TFRecord example notebook
// Run with Scala shell or in a Scala notebook

// Spark 3.x and Scala 2.12
// Note, consider using YARN or K8S to scale out
JAR=http://canali.web.cern.ch/res/spark-tensorflow-connector_2.12-1.11.0.jar
bin/spark-shell --master local[*] --driver-memory 20g --jars $JAR

// spark 2.4.8 and scala 2.11
// bin/spark-shell --master local[*] --driver-memory 20g --packages org.tensorflow:spark-tensorflow-connector_2.11:1.14.0

// UDF to convert Vectors to Arrays
// This is needed as TFRecord cannot handle Vector Type, but can save Arrays
import org.apache.spark.ml.linalg.Vector
val toArray = udf { v: Vector => v.toArray }
spark.udf.register("toArray", toArray)

// Data input and output Paths
val PATH = "<..EDIT_PATH..>/Data/"
val outputPath = PATH
val numPartitions = 200
val df=spark.read.parquet(PATH + "testUndersampled.parquet")

scala> df.printSchema
root
 |-- hfeatures: vector (nullable = true)
 |-- label: long (nullable = true)
 |-- lfeatures: array (nullable = true)
 |    |-- element: array (containsNull = true)
 |    |    |-- element: double (containsNull = true)
 |-- hfeatures_dense: vector (nullable = true)
 |-- encoded_label: vector (nullable = true)
 |-- HLF_input: vector (nullable = true)
 |-- GRU_input: array (nullable = true)
 |    |-- element: array (containsNull = true)
 |    |    |-- element: double (containsNull = true)

// Save the test dataset in TFRecord format
// Select the fields used by the Inclusive classifier: HLF_input, GRU_input and encoded_label
// Note: GRU_input is flattened, this allows to use the Example record format (default),
// restoring the Array to its original shape will be handled in TensorFlow using tf.data and tf.io
//
df.coalesce(numPartitions).
selectExpr("toArray(HLF_input) as HLF_input", "flatten(GRU_input) as GRU_input", "toArray(encoded_label) as encoded_label")
.write.format("tfrecords")
.save(outputPath+"testUndersampled.tfrecord")

//
// Repeat for the training dataset
//

// Read Parquet
val df2=spark.read.parquet(PATH + "trainUndersampled.parquet")

// save the training dataset in TFRecord format
// compact output in numPartitions files with coalesce(numPartitions)
//
df2.coalesce(numPartitions).
selectExpr("toArray(HLF_input) as HLF_input", "flatten(GRU_input) as GRU_input", "toArray(encoded_label) as encoded_label")
.write.format("tfrecords")
.save(outputPath+"trainUndersampled.tfrecord")
