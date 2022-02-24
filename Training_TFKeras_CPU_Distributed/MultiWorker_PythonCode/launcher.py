#########
# launcher.py - a basic wrapper to lauch parallel training using tf.keras and tf.distribute
# use: python launcher.py
#
# Configure: set the list of endpoints and the script name
# Limitations: this script currently only parallelizes training on a single node.
#              TODO: extend to run parallel training on multiple nodes.
# 
# tested with tf 2.0.0-rc0
########

import os
from subprocess import Popen

## Configuration
#    - configure the list of endpoints, this defines the parallel degree for the training
#    - define the name of the training script

nodes_endpoints = "localhost:12345, localhost:12346, localhost:12347, localhost:12348" # will train with 4 workers 
training_script_name="4.3a_InclusiveClassifier_WorkerCode.py"         # training script

# This envirment variable is used as a way to pass the parameter to the training script
os.environ["NODES_ENDPOINTS"]=nodes_endpoints

print("Launcher for distributed training of " + training_script_name)
print("node endpoints: " + nodes_endpoints)

process_list = []
tot_workers = len(nodes_endpoints.split(","))
for i in range(tot_workers):
    # each worker has a unique worker number, we pass it via this environment vaiable
    os.environ["WORKER_NUMBER"] = str(i)
    # run the training script asynchronously
    with open(f"worker{i}-stdout.txt", "w") as fout:
        with open(f"worker{i}-stderr.txt", "w") as ferr:
            process_list.append(Popen(["python", training_script_name], stdout=fout, stderr=ferr))
    print("..launched worker " + str(i))

print("All workers launched")

print("Waiting for training processes to finish")
exit_codes = [p.wait() for p in process_list]

print("All training processes terminated.")
exit

