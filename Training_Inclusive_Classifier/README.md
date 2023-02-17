## Training the inclusive classifier

This directory contains notebooks and code for training the inclusive  classifier.  
This contains the 14 High-Level Features classifier and notably in addition to that also an array of up to 801 particles with 19 features each.
The latter is responsible for the large size of the training dataset (250 GB) and is handled using methods like LSTM and GRU, suitable for sequences of data.    
The input are 14 features, described in Topology classification with deep learning to improve real-time event selection at the LHC.  
The output are 3 classes, "W + jet", "QCD", "t tbar", see also Machine Learning Pipelines with Modern Big Data Tools for High Energy Physics Comput Softw Big Sci 4, 8 (2020).  
Training on GPU has been used to speed up execution on these notebooks.  

The techniques tested include:
- [TensorFlow running on GPU, GRU-based model, data in TFRecord](4.3a-Training-InclusiveClassifier-GRU-TF_Keras_TFRecord.ipynb)
- [TensorFlow running on GPU, LSTM-based model, data in TFRecord](4.3a-Training-InclusiveClassifier-LSTM-TF_Keras_TFRecord.ipynb)
