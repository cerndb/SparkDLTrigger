## Training the HLF classifier

This directory contains notebooks and code for training the High-Level Features (HLF) particle classifier. 
The HLF classifier is designed to classify labeled data using 14 input features, as described in the paper
"Topology classification with deep learning to improve real-time event selection at the LHC". 
The classifier aims to categorize events into three classes: "W + jet", "QCD", and "t tbar". 
For further details, please refer to the publication 
"Machine Learning Pipelines with Modern Big Data Tools for High Energy Physics" (Comput Softw Big Sci 4, 8, 2020).

The provided notebooks demonstrate how to feed data in Parquet format to TensorFlow/Keras for training. 
Additionally, GPU acceleration has been utilized to enhance the execution speed of these notebooks. 
Some of the techniques explored and tested in this training process include:

- Model architecture design and configuration
- Data preprocessing and normalization
- Model compilation and training
- Evaluation and performance metrics
- Model visualization and analysis

By following the notebooks in this directory, you can gain insights into the training methodology and 
techniques employed to achieve accurate classification results using the HLF classifier. 
It serves as a resource for understanding the practical implementation of deep learning models in 
high-energy physics and event classification.

Please refer to the specific notebooks for detailed instructions, code examples, and visualizations related
to each technique.

## Contents

## Deep Learning and basic Data pipelines
These notebooks provide examples of how to integrate Deep Learning frameworks with some basic data pipelines using Pandas to feed data into the DL training step.  
They implement a  Particle classifier using different DL frameworks. The data is stored in Parquet format, which is a columnar format that is very efficient for reading data,
it processed using Pandas, and then fed into the DL training step.

* [TensorFlow classifier with PySpark](TensorFlow_Keras_HLF_with_PySpark_Parquet.ipynb)
* [TensorFlow classifier with PyArrow](TensorFlow_Keras_HLF_with_PyArrow_Parquet.ipynb)
* [TensorFlow classifier with Pandas](TensorFlow_Keras_HLF_with_Pandas_Parquet.ipynb)
* [Pytorch classifier with Pandas](PyTorch_HLF_with_Pandas_Parquet.ipynb)
* [Pytorch Lightning classifier with Pandas](PyTorch_Lightning_HLF_with_Pandas_Parquet.ipynb)
* [XGBoost classifier with Pandas](XGBoost_with_Pandas_Parquet.ipynb)

## More advanced Data pipelines
These examples show some more advanced data pipelines, useful for training with large data sets. They show how to use
the Petastorm library to read data from Parquet files with TensorFlow and PyTorch, and how to use the TFRecord format with TensorFlow.

* [TensorFLow and Petastorm](TensorFlow_Keras_HLF_with_Petastorm_Parquet.ipynb)
* [PyTorch and Petastorm](PyTorch_HLF_with_Petastorm_Parquet.ipynb)
* [TensorFlow with TFRecord](TensorFlow_Keras_HLF_with_TFRecord.ipynb)
