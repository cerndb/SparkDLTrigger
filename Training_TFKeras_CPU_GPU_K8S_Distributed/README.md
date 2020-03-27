## Training using tf.keras and tf.distribute parallelized on Kubernetes clusters for CPU and GPU training

This folder contains code used for training the Inclusive Classifier with tf.keras in distributed mode using a Kubernetes cluster.
It is intended to be used together with the tool [tf-spawner](https://github.com/cerndb/tf-spawner)
Model training scripts and files:
- `4.3a_InclusiveClassifier_WorkerCode_tuned_cached_learningAdaptive.py`
- `4.3a_InclusiveClassifier_WorkerCode_tuned_NOcached_learningAdaptive.py` (this version does not cache the dataset, use when not enough memory is available)
- `4.3a_InclusiveClassifier_WorkerCode_tuned_cached_shuffle_learningAdaptive.py` (this version adds shuffling of the training dataset at each epoch)
- `pod-cpu.yaml` and `pod-gpu.yaml`: these are used by tf-spawner (see exmaple below) 

**How to use:**
- Download TF-Spawner: `git clone https://github.com/cerndb/tf-spawner`
- Install the dependencies: `pip3 install kubernetes` 
- Set up you Kubernetes environment (if needed): `export KUBECONFIG=<path_to_kubectl config file`
- Copy the data to the cloud service, for example to a S3-compatible filesystem, and set the required environment variables,
 for example, edit the file `examples/envfile.example` with the
  ```
  S3_ENDPOINT=...
  AWS_ACCESS_KEY_ID=...
  AWS_SECRET_ACCESS_KEY=...
  AWS_LOG_LEVEL=3
  ```
- Edit the configurable variables in the training script 
  - notably, edit `PATH="s3://sparktfdata/"` to point to the location where you have copied the data.

**Run distributed training on Kubernets with TF-Spawner**
 
- Run on CPU as in this example (with 10 workers):
  - `./tf-spawner -w 10 -i tensorflow/tensorflow:2.0.1-py3 -e examples/envfile.example --pod-file <PATH>/pod-cpu.yaml <PATH>/4.3a_InclusiveClassifier_WorkerCode_tuned_cached_learningAdaptive.py`

- When training using GPU resources on Kubernetes start with this example (with 10 GPU nodes):
  - `./tf-spawner -w 10 -i tensorflow/tensorflow:2.0.1-gpu-py3 --pod-file <PATH>/pod-gpu.yaml -e examples/envfile.example <PATH>/4.3a_InclusiveClassifier_WorkerCode_tuned_cached_learningAdaptive.py`

**Notes:**
The training scripts provided in this folder have been tested on a Kubernetes cluster with CPU and GPU resources, using TF 2.0.1.  
Notable TensorFlow modules used by the training script:
- tf.keras for model definition
- tf.distribute strategy with MultiWorkerMirroredStrategy to parallelize the training.
- tf.data to read the training and test dataset, from files in TFRecord format (see also [data folder](../Data)).
