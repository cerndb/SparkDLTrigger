## Training the Inclusive Classifier

The Inclusive Classifier is the more advanced model (compared to the simplified High Level Features Classifier).  
For each event, the Inclusive Classifier uses the 14 High-Level Features classifier with the addition of a list of up to 801 particles with 19 features each.
The latter is responsible for the large size of the training dataset (250 GB) and is handled using recursive neural netoworks, 
implemented with LSTM or GRU layers that are suitable for processing data sequences.      
The model output are 3 classes, "W + jet", "QCD", "t tbar", see also 
[Machine Learning Pipelines with Modern Big Data Tools for High Energy Physics Comput Softw Big Sci 4, 8 (2020)](https://rdcu.be/b4Wk9).    
Training on GPU has been used to speed up execution on these notebooks.  

Notebooks and the techniques tested:
- **[Data in TFRecord, TensorFlow on GPU, GRU-based model](4.3a-Training-InclusiveClassifier-GRU-TF_Keras_TFRecord.ipynb)**
- **[Data in Parquet read with Petastorm, TensorFlow on GPU, GRU-based model](4.3a-Training-InclusiveClassifier-GRU-TF_Keras_Parquet_Petastorm.ipynb)**
- **[Data in TFRecord, TensorFlow on GPU, LSTM-based model](4.3a-Training-InclusiveClassifier-LSTM-TF_Keras_TFRecord.ipynb)**
