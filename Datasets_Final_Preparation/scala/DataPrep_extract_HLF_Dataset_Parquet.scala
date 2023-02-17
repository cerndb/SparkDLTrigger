//
// Extract the High Level features training and test datasets from the full datasets
// - Convert Vectors into Arrays
// - Save the resulting Spark DataFrames as Parquet files
// This is used as input for the TensorFlow and Petastorm example notebooks
// Run with Scala shell or in a Scala notebook

// Define a UDF to transform Vectors in Arrays
// This is because we need Array for the Petastorm example notebook

import org.apache.spark.ml.linalg.Vector
val toArray = udf { v: Vector => v.toArray }
spark.udf.register("toArray", toArray)

// Data source

val PATH = "hdfs://analytix/Training/Spark/TopologyClassifier/"
val outputPATH = PATH
val df=spark.read.parquet(PATH + "testUndersampled.parquet").select("HLF_input", "encoded_label")

scala> df.printSchema
root
 |-- HLF_input: vector (nullable = true)
 |-- encoded_label: vector (nullable = true)

// Save the test dataset
// Compact output in 1 file with coalesce(1)
// Additional (optional) tuning is to set the Parquet block size of 1MB, this forces row groups to 1MB. 
// This action is motivated by the use of Petastorm. 
// Petastorm uses Parquet block size in  make_batch_reader to determine the batch size to feed to Tensorflow.
// If you don't need to use Petastorm, you can skip the setting option("parquet.block.size", 1024 * 1024) and use defaults
//
df.selectExpr("toArray(HLF_input) as HLF_input", "toArray(encoded_label) as encoded_label").
  coalesce(1).
  write.  
  option("parquet.block.size", 1024 * 1024).  
  parquet(outputPATH + "testUndersampled_HLF_features.parquet")

//
// Repeat for the training dataset
//

val df2=spark.read.parquet(PATH + "trainUndersampled.parquet")

df2.selectExpr("toArray(HLF_input) as HLF_input", "toArray(encoded_label) as encoded_label").
  coalesce(4).write.
  option("parquet.block.size", 1024 * 1024).
  parquet(outputPATH + "trainUndersampled_HLF_features.parquet")
