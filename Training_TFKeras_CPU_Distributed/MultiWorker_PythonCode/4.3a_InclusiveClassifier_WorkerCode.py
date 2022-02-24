#########
# This Python code is the core of the distributed training with 
# tf.keras with tf.distribute for the Inclusive Classifier model
# 
# tested with TensorFlow 2.0.0-rc0
#########

########################
## Configuration

import os
nodes_endpoints_raw = os.environ.get("NODES_ENDPOINTS")  # example: "localhost:12345, localhost:12346, localhost:12347"
nodes_endpoints = [nodename.strip() for nodename in nodes_endpoints_raw.split(",")]
number_workers = len(nodes_endpoints)
worker_number = int(os.environ.get("WORKER_NUMBER")) # example: 0

# data paths
PATH="/local3/lucatests/Data/" # training and test data parent dir
model_output_path="./"         # output dir for saving the trained model 

# tunables
batch_size = 128 * number_workers
validation_batch_size = 10240

# tunable
num_epochs = 12

## End of configuration
########################

## Main code for the distributed model training

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Masking, Dense, Activation, GRU, Dropout, concatenate
import os
import json

# TF_CONFIG is the envirnment variable used to configure tf.distribute
# each worker will have a different number, an index entry in the nodes_endpoint list
# node 0 is master, by default
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': nodes_endpoints
    },
    'task': {'type': 'worker', 'index': worker_number}
})

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
    optimizer = 'Adam'
    loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"] )

# test dataset 
files_test_dataset = tf.data.Dataset.list_files(PATH + "testUndersampled.tfrecord/part-r*", shuffle=False)
# training dataset 
files_train_dataset = tf.data.Dataset.list_files(PATH + "trainUndersampled.tfrecord/part-r*", seed=4242)

# tunable
num_parallel_reads=tf.data.experimental.AUTOTUNE # TF2.0
# num_parallel_reads=8

test_dataset = files_test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).interleave(
    tf.data.TFRecordDataset, 
    cycle_length=num_parallel_reads,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_dataset = files_train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).interleave(
    tf.data.TFRecordDataset, cycle_length=num_parallel_reads,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

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

parsed_test_dataset=test_dataset.map(decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
parsed_train_dataset=train_dataset.map(decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train=parsed_train_dataset.repeat().batch(batch_size)

num_train_samples=3426083   # there are 3426083 samples in the training dataset
steps_per_epoch=num_train_samples//batch_size

test=parsed_test_dataset.repeat().batch(validation_batch_size)

num_test_samples=856090 # there are 856090 samples in the test dataset
validation_steps=num_test_samples//validation_batch_size  


callbacks = [ tf.keras.callbacks.TensorBoard(log_dir='./logs') ]
# callbacks = []
    
history = model.fit(train, steps_per_epoch=steps_per_epoch, \
                    validation_data=test, validation_steps=validation_steps, \
                    epochs=num_epochs, callbacks=callbacks, verbose=1)

model_full_path=model_output_path + "mymodel" + str(worker_number) + ".h5"
print("Training finished, now saving the model in h5 format to: " + model_full_path)

model.save(model_full_path, save_format="h5")

# TensorFlow 2.0
model_full_path=model_output_path + "mymodel" + str(worker_number) + ".tf"
print("..saving the model in tf format (TF 2.0) to: " + model_full_path)
tf.keras.models.save_model(model, PATH+"mymodel" + ".tf", save_format='tf')

exit

