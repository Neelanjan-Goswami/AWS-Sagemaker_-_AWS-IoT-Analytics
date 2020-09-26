import numpy as np
import os
import tensorflow as tf

INPUT_TENSOR_NAME = 'inputs'

# Disable MKL to get a better perfomance for this model.
#os.environ['TF_DISABLE_MKL'] = '1'
#os.environ['TF_DISABLE_POOL_ALLOCATOR'] = '1'


def estimator_fn(run_config, params): #Function 1 Estimator 
    feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[4])] #shape represents the four elements of the data set 
    return tf.estimator.DNNClassifier(feature_columns=feature_columns,#s
                                      hidden_units=[10, 20, 10], #Deep Neural Network Classifier
                                      n_classes=3, #Three output classes because we have three species of flowers
                                      config=run_config)


def serving_input_fn(params):  #Function 2 Once the model is deployed how model is going to get the inputs 
    feature_spec = {INPUT_TENSOR_NAME: tf.FixedLenFeature(dtype=tf.float32, shape=[4])}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()


def train_input_fn(training_dir, params): #Function 3 Here we are loading the training data--This is the fucntion sagemaker going to cal during training process
    """Returns input function that would feed the model during training"""
    return _generate_input_fn(training_dir, 'iris_training.csv')


def eval_input_fn(training_dir, params):#Function 4 Evaluation data--Basically data is split into training data and evaluation data.
    """Returns input function that would feed the model during evaluation"""
    return _generate_input_fn(training_dir, 'iris_test.csv')


def _generate_input_fn(training_dir, training_filename): #Function 5 train_input_fn and eval_input_fnv is calling _generate_input_fn
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=os.path.join(training_dir, training_filename),
        target_dtype=np.int,
        features_dtype=np.float32)

    return tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)()
