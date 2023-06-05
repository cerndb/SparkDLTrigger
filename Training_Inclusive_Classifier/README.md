## Training the Inclusive Classifier

The Inclusive Classifier represents a more advanced model compared to the simplified 
High-Level Features Classifier. It incorporates the 14 High-Level Features classifier and expands it
with a list of up to 801 particles, each characterized by 19 features. 
As a result, the training dataset is significantly larger (250 GB), requiring the utilization of
recursive neural networks. LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Unit) layers are
employed to process data sequences efficiently.

The model's objective remains the same, aiming to classify events into three classes: "W + jet", "QCD",
and "t tbar". For further details, please refer to the publication 
"Machine Learning Pipelines with Modern Big Data Tools for High Energy Physics" (Comput Softw Big Sci 4, 8, 2020).

The notebooks provided in this directory demonstrate various techniques and approaches for training 
the Inclusive Classifier. They cover the following scenarios:

- **[Data in TFRecord, TensorFlow on GPU, GRU-based model](TensorFlow_Inclusive_Classifier_GRU_TFRecord.ipynb)**
  - Description: This notebook showcases the training process using data stored in TFRecord format.
  - TensorFlow is configured to run on a GPU, and a GRU-based model architecture is employed.

- **[Data in Parquet read with Petastorm, TensorFlow on GPU, GRU-based model](TensorFlow_Inclusive_Classifier_GRU_TFRecord.ipynb)**
   - Description: This notebook explores the usage of Parquet data, read with Petastorm, for training.
   - TensorFlow is configured to run on a GPU, and a GRU-based model architecture is utilized.

- **[Data in TFRecord, TensorFlow on GPU, LSTM-based model](TensorFlow_Inclusive_Classifier_LSTM_TFRecord.ipynb)**
   - Description: This notebook focuses on training with data stored in TFRecord format. 
   - TensorFlow is configured to run on a GPU, and an LSTM-based model architecture is employed.

- **[Data in TFRecord, TensorFlow on GPU, Transformer-based model](TensorFlow_Inclusive_Classifier_Transformer_TFRecord.ipynb)**
  - Description: This notebook focuses on training with data stored in TFRecord format.
  - TensorFlow is configured to run on a GPU, and a Transformer-based model architecture is employed.
