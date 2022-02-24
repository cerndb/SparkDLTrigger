This folder contains code for training the Inclusive classifier with tf.keras in distributed mode using Python scripts
tf.distrubute strategy with MultiWorkerMirroredStrategy is used to parallelize the training.
tf.data is used to read the data in TFRecord format.

- launcher.py - configure the distributed training environment and runs parallel training. 
  - experiment changing the number of training instances 
  - this script is basic, currently limited to parallelize inside one single node
  - TODO: extend to run on multiple nodes 
- Load and display metrics of the resulting model using the notebook 4.3a-Model_evaluate_ROC_and_CM.ipynb


