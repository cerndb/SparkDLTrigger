This folder contains code for training the Inclusive classifier with tf.keras in distributed mode using notebooks.
tf.distrubute strategy with MultiWorkerMirroredStrategy is used to parallelize the training.
tf.data is used to read the data in TFRecord format.

- Run all worker notebooks concurrently (you'll find that tf.distribute will block waiting for all workers defined in TF_CONFIG are running)
 - 4.3a-Worker0_Training-InclusiveClassifier-TF_Keras_TFRecord.ipynb will be the master node but it will not start training till the rest of the notebooks are active
 - experiment changing the number of training instances, you can also distribute training over multiple nodes with minor changes in the values of TF_CONFIG
- Load and display metrics of the resulting model using the notebook 4.3a-Model_evaluate_ROC_and_CM.ipynb


