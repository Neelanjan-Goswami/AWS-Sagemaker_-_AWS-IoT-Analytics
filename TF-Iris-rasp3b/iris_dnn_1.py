from __future__ import absolute_import

import numpy as np
import os
import tensorflow as tf

INPUT_TENSOR_NAME = 'inputs'
input_tensor_name = 'inputs'

def estimator_fn(run_config, params): #Function 1 Estimator 
    feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[4])] #shape represents the four elements of the data set 
    return tf.estimator.DNNClassifier(feature_columns=feature_columns,#s
                                      hidden_units=[10, 20, 10], #Deep Neural Network Classifier
                                      n_classes=3, #Three output classes because we have three species of flowers
                                      config=run_config)


def serving_input_fn(params):  #Function 2 Once the model is deployed how model is going to get the inputs 
    feature_spec = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32,[None,4],name=None)}
    return tf.estimator.export.ServingInputReceiver(feature_spec,feature_spec)


def train_input_fn(training_dir, params):
    """Returns input function that would feed the model during training"""
    return _generate_input_fn(training_dir, "iris_training.csv", params)


def eval_input_fn(training_dir, params):
    """Returns input function that would feed the model during evaluation"""
    return _generate_input_fn(training_dir, "iris_test.csv", params)


def _generate_input_fn(training_dir, training_filename, params):

    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=os.path.join(training_dir, training_filename),
        target_dtype=np.int,
        features_dtype=np.float32,
    )

    return tf.estimator.inputs.numpy_input_fn(
        x={input_tensor_name: np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True,
    )()