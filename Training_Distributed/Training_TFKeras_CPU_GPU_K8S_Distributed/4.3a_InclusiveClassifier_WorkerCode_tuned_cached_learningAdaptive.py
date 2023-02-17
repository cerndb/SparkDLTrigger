#########
# This Python code is the core of the distributed training with 
# tf.keras with tf.distribute for the Inclusive Classifier model
# Intended to be used with tf-spawner: https://github.com/cerndb/tf-spawner
#
# tested with TensorFlow 2.0.1 and 2.0.1-gpu
#########

########################
## Configuration

import os
worker_number = int(os.environ.get("WORKER_NUMBER")) # example: 0
number_workers = int(os.environ["TOT_WORKERS"])

# data paths
PATH="s3://sparktfdata/" # training and test data parent dir
model_output_path="./"         # output dir for saving the trained model 

# tunables
batch_size = 128 * number_workers
validation_batch_size = 1024

# tunable
num_epochs = 6

## End of configuration
########################

## Main code for the distributed model training

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Masking, Dense, Activation, GRU, Dropout, concatenate,LSTM
from time import time

# TF_CONFIG is the environment variable used to configure tf.distribute
# each worker will have a different number, an index entry in the nodes_endpoint list
# node 0 is master, by default
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# This implements the distributed stratedy for model
with strategy.scope():
    ## GRU branch
    gru_input = Input(shape=(801,19), name='gru_input')
    a = gru_input
    a = Masking(mask_value=0.)(a)
    a = GRU(units=50,activation='tanh')(a)
    gruBranch = Dropout(0.2)(a)
    
    hlf_input = Input(shape=(14), name='hlf_input')
    b = hlf_input
    hlfBranch = Dropout(0.2)(b)

    c = concatenate([gruBranch, hlfBranch])
    c = Dense(25, activation='relu')(c)
    output = Dense(3, activation='softmax')(c)
    
    model = Model(inputs=[gru_input, hlf_input], outputs=output)
    
    ## Compile model
    optimizer = Adam(learning_rate=0.0005*number_workers)
    loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"] )

# test dataset 
files_test_dataset = tf.data.Dataset.list_files(PATH + "testUndersampled.tfrecord/part-r-0*", shuffle=False)
# training dataset 
files_train_dataset = tf.data.Dataset.list_files(PATH + "trainUndersampled.tfrecord/part-r-0*", seed=4242)

# tunable
num_parallel_reads=128
num_par_calls=128
buf_size=2**30

test_dataset = files_test_dataset.prefetch(buffer_size=buf_size).interleave(
    tf.data.TFRecordDataset, 
    cycle_length=num_parallel_reads,
    num_parallel_calls=num_par_calls)

train_dataset = files_train_dataset.prefetch(buffer_size=buf_size).interleave(
    tf.data.TFRecordDataset, cycle_length=num_parallel_reads,
    num_parallel_calls=num_par_calls)

# Function to decode TF records into the required features and labels
def decode(serialized_example):
    deser_features = tf.io.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'HLF_input': tf.io.FixedLenFeature((14), tf.float32),
          'GRU_input': tf.io.FixedLenFeature((801,19), tf.float32),
          'encoded_label': tf.io.FixedLenFeature((3), tf.float32),
          })
    return((deser_features['GRU_input'], deser_features['HLF_input']), deser_features['encoded_label'])

parsed_test_dataset=test_dataset.map(decode, num_parallel_calls=num_par_calls)
parsed_train_dataset=train_dataset.map(decode, num_parallel_calls=num_par_calls).cache()

train=parsed_train_dataset.repeat().batch(batch_size)

num_train_samples=3426083   # there are 3426083 samples in the training dataset
steps_per_epoch=num_train_samples//batch_size

test=parsed_test_dataset.repeat().batch(validation_batch_size)
num_test_samples=856090 # there are 856090 samples in the test dataset
validation_steps=num_test_samples//validation_batch_size  

def scheduler(epoch):
  if epoch <= 2:
    return 0.0005 * strategy.num_replicas_in_sync
  elif epoch <=4:
    return 0.0002 * strategy.num_replicas_in_sync
  else:
    return 0.0001 * strategy.num_replicas_in_sync

#Add this for TensorBoard
#callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs', profile_batch = 0)]
callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler)]
    
print('starting at ', time())
history = model.fit(train, steps_per_epoch=steps_per_epoch, \
                    epochs=num_epochs, callbacks=callbacks, verbose=1)

print("finishing at:", time())
print("evaluating:", time())
model.evaluate(test, steps=validation_steps)
print("evaluated at:", time())

model_full_path= "/tmp/mymodel" + str(worker_number) + ".h5"
print("Training finished, now saving the model in h5 format to: " + model_full_path)
model.save(model_full_path, save_format="h5")
print("model saved.\n")

#print("..saving the model in tf format (TF 2.0) to: " + model_full_path)
#tf.keras.models.save_model(model, "/tmp/mymodel"+ str(worker_number) + ".tf", save_format='tf')
#print("model saved.\n")

exit()


