## Distributed training

This folder contains code for training the Inclusive classifier with tf.keras in distributed mode.
tf.distribute strategy with MultiWorkerMirroredStrategy is used to parallelize the training.
tf.data is used to read the data in TFRecord format.

- [MultiWorker_Notebooks](MultiWorker_Notebooks): distributed training and model performance metrics visualization using notebooks
- [MultiWorker_PythonCode](MultiWorker_PythonCode): distributed training using "manual" Python code, suitable for local node testing
- [Training_TFKeras_CPU_GPU_K8S_Distributed](Training_TFKeras_CPU_GPU_K8S_Distributed): Distributed training using Kubertes and a custom tool TF_Spawner.

**Note:** see also [Training_TFKeras_CPU_GPU_K8S_Distributed](../Training_TFKeras_CPU_GPU_K8S_Distributed) for
distributed training on Kubernets clusters, using the custom TF-Spawner tool 

