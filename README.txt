Sequential skip prediction challenge

Requirements for training a neural network:
keras 2.0.6
tensorflow 1.0.0
numpy 1.15.4
pandas 0.23.4
matplotlib 2.0.2

Additional requirements for preprocessing the data:
joblib 0.11


The code structure should be the following:
-code
---neural_nets
---preprocessing
-data
---training_set
---training_set_preproc
---test_set
---test_set_preproc


A small share of the data set (including preprocessed data) can be found here:
https://polybox.ethz.ch/index.php/s/oAGcCUzedwvCef1
The full data set can be found here:
https://www.crowdai.org/challenges/spotify-sequential-skip-prediction-challenge/dataset_files


Running the preprocessing:
For preprocessing the training files the following command has to be used

python code/preprocessing/main_preprocessing.py -r 1

and for the testing set

python code/preprocessing/main_preprocessing.py -t 1

The files will then be written into the directory training_set_preproc resp. test_set_preproc


Running the neural networks:
The main file uses input arguments to specify the the desired tasks:
To create a neural network locally and train it with the log_min.csv data set:

python code/neural_nets/main.py -q 0

The further tasks, which can be started, are listed below. Locally means with a
small data set for running commands on a laptop, while on server is supposed to run on the Leonhard server.

-create a network and use a generator to load the data locally: -q 3
-run a grid search on different models locally: -q 4
-create a simple model on server: -q 10
-load and continue to train a model on the server: -q 11
-load a model and predict the private test set of the challenge on the server: -q 12
-create a network and use a generator to load the data on the server: -q 13
-run a grid search on different models on the server: -q 14
