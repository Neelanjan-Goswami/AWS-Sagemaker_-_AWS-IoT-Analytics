import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.estimator.model_fn import ModeKeys as Modes

import argparse
'''
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    #parser.add_argument('--epochs', type=int, default=10)
    #parser.add_argument('--batch_size', type=int, default=100)
    #parser.add_argument('--learning_rate', type=float, default=0.1)

    # input data and model directories
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_known_args()
'''

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_dir', type=str)
    args = parser.parse_args()

    
    


def estimator_fn(params):
    feature_columns = [tf.feature_column.numeric_column(key="sepal_length", dtype=tf.float32),
                          tf.feature_column.numeric_column(key="sepal_width", dtype=tf.float32),
                          tf.feature_column.numeric_column(key="petal_length", dtype=tf.float32),
                          tf.feature_column.numeric_column(key="petal_width", dtype=tf.float32)]

    estimator =  tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3)

    estimator.train(input_fn=input_fn_train, steps=1000)

    
    estimator.export_saved_model(model_dir, serving_input_receiver_fn=serving_input_receiver_fn)
'''

# classifier = tf.estimator.Estimator(
#     model_fn=model_fn,
#     params={
#         'feature_columns': my_feature_columns,
#         # Two hidden layers of 10 nodes each.
#         'hidden_units': [10, 10],
#         # The model must choose between 3 classes.
#         'n_classes': 3,
#     })




INPUT_TENSOR_NAME="inputs"

def model_fn(features, labels, mode, params):
    
    net = tf.reshape(features, [-1,4])
    #net = tf.feature_column.input_layer(features, [-1,4])
  # Connect the first hidden layer to input layer
  # (features["x"]) with relu activation
    dense1 = tf.layers.dense(inputs=net, units=10, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=20, activation=tf.nn.relu)
    dense3 = tf.layers.dense(inputs=dense2, units=10, activation=tf.nn.relu)

  # Connect the output layer to second hidden layer (no activation fn)
    logits = tf.layers.dense(dense3, units=3, activation=None)

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            #'logits': logits,
        }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    accuracy = tf.metrics.accuracy(labels=labels,
                               predictions=predicted_classes,
                               name='acc_op')
    
    metrics = {'accuracy': accuracy}

 
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)
    
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)








def parser(record):
    features = {
   'sepal_length': tf.FixedLenFeature([], tf.float32),
   'sepal_width': tf.FixedLenFeature([], tf.float32),
   'petal_length': tf.FixedLenFeature([], tf.float32),
   'sepal_width': tf.FixedLenFeature([], tf.float32),
   'label': tf.FixedLenFeature([], tf.int64)
   }
    
    return tf.parse_single_example(record, features)

def serving_input_receiver_fn(params):
  #An input receiver that expects a serialized tf.Example."""
    features = {
   'sepal_length': tf.FixedLenFeature([], tf.float32),
   'sepal_width': tf.FixedLenFeature([], tf.float32),
   'petal_length': tf.FixedLenFeature([], tf.float32),
   'sepal_width': tf.FixedLenFeature([], tf.float32),
   'label': tf.FixedLenFeature([], tf.int64)
    }
    
    serialized_tf_example = {
        'sepal_length': tf.placeholder(tf.float32, [None, 1]),
        'sepal_width': tf.placeholder(tf.float32, [None, 1]),
        'petal_length': tf.placeholder(tf.float32, [None, 1]),
        'petal_width': tf.placeholder(tf.float32, [None, 1]),
    }
    receiver_tensors = {'feats': serialized_tf_example}
    features_spec = tf.parse_example(serialized_tf_example, features)
    return tf.estimator.export.ServingInputReceiver(features_spec, receiver_tensors)



def train_input_fn(training_dir, params):
    print("training_dir",training_dir)
    return _input_fn(training_dir, 'train.tfrecords', batch_size=1)


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'test.tfrecords', batch_size=1)


def _input_fn(tfrecords_path , training_filename , batch_size=1):

    test_file = os.path.join(tfrecords_path, training_filename)
    dataset = (
    tf.data.TFRecordDataset(test_file)
    .map(parser)
    .batch(1)
    )
  
    iterator = dataset.make_one_shot_iterator()

    features = iterator.get_next()
    labels = iterator.get_next()
    return features, labels
    #return {INPUT_TENSOR_NAME: batch_feats}, batch_labels


'''
def parser(record):

  features={
    'feats': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64),
  }
  
  parsed = tf.parse_single_example(record, features)
  feats = tf.convert_to_tensor(tf.decode_raw(parsed['feats'], tf.float64))
  label = tf.cast(parsed['label'], tf.int32)

  return {'feats': feats}, label


def my_input_fn(tfrecords_path):

  dataset = (
    tf.data.TFRecordDataset(tfrecords_path)
    .map(parser)
    .batch(1)
  )
  
  iterator = dataset.make_one_shot_iterator()

  batch_feats, batch_labels = iterator.get_next()


  return batch_feats, batch_labels
  '''