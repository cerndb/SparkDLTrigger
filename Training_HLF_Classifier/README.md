## Training the HLF classifier

This directory contains notebooks and code for training the High Level Features particle classifier.
The High-Level Features classifier is built with labeled data.
The input are 14 features, described in Topology classification with deep learning to improve real-time event selection at the LHC.
The output are 3 classes, "W + jet", "QCD", "t tbar", see also Machine Learning Pipelines with Modern Big Data Tools for High Energy Physics Comput Softw Big Sci 4, 8 (2020)

This also shows how to feed data in Parquet format to TensorFlow/Keras.  
Training on GPU has been used to speed up execution on these notebooks.  
The techniques tested include:
- TensorFlow: Read Parquet in numpy arrays and feed those from memory to TensorFlow
  - [Read with Parquet using PySpark and feed data from memory to tf.keras](4.0c-Training-HLF-TF_Keras_PySpark_Parquet.ipynb)
  - [Read with Pandas](4.0c_bis-Training-HLF-TF_Keras_Pandas_Parquet.ipynb)
  - [Read with PyArrow](4.0c_tris-Training-HLF-TF_Keras_Pyarrow_Parquet.ipynb)
- TensorFlow: Petastorm is used to read Parquet and feed it to TensorFlow
  - [tf.keras, Parquet data, and Petastorm](4.0d-Training-HLF-TF_Keras_Petastorm_Parquet.ipynb)
- TensorFlow using tf.data and datasets in TFRecord format to work natively with data in 
  - [tf.data and the TFRecord format](4.0e-Training-HLF-TF_Keras_TFRecord.ipynb)
- XGBoost
  - [4.0f-Training-HLF-XGBoost_Pandas_Parquet.ipynb](4.0f-Training-HLF-XGBoost_Pandas_Parquet.ipynb)
