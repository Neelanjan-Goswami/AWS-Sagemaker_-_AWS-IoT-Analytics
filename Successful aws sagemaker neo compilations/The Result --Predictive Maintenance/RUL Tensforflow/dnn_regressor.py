import os
import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes

INPUT_TENSOR_NAME = 'inputs'
SIGNATURE_NAME = 'predictions'
LEARNING_RATE = 0.001

from tensorflow.python.estimator.model_fn import ModeKeys as Modes

def model_fn(features, labels, mode, params):
    # Input Layer


#     input_layer = tf.reshape(features[INPUT_TENSOR_NAME],[-1,26])

#     dense1 = tf.layers.dense(inputs=input_layer, units=128, activation=tf.nn.relu)
# #    drop1 = tf.layers.dropout(dense1, rate=0.5)
#     dense2 = tf.layers.dense(inputs=dense1, units=64, activation=tf.nn.relu)
# #    drop2 = tf.layers.dropout(dense2, rate=0.5)
#     dense3 = tf.layers.dense(inputs=dense2, units=32, activation=tf.nn.relu)
# #     drop3 = tf.layers.dropout(dense3, rate=0.5)
#     dense4 = tf.layers.dense(inputs=dense3, units=16, activation=tf.nn.relu)
# #     drop4 = tf.layers.dropout(dense3, rate=0.5)
#     logits = tf.layers.dense(inputs=dense4, units=1, activation=tf.nn.relu)
    batch_size = 1
#     num_iterations = 1000
    timesteps = 5
    element_size = 24
    num_classes = 1
    hidden_layer_size = 128
    _inputs = tf.reshape(features[INPUT_TENSOR_NAME], [-1,timesteps,element_size])
#     y = tf.placeholder(tf.float32, shape=[num_classes],name='inputs')
#     _inputs = tf.placeholder(tf.float32,shape=[1, timesteps,element_size],name='inputs')
#     y = tf.placeholder(tf.float32, shape=[1, num_classes],name='inputs')
    

    # TensorFlow built-in functions
    rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
    outputs, _ = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)
    Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes],mean=0,stddev=.01))
    bl = tf.Variable(tf.truncated_normal([num_classes],mean=0,stddev=.01))
    last_rnn_output = outputs[:,-1,:]
    final_output = tf.matmul(last_rnn_output, Wl) + bl
#     softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output,labels=y)

    
    

    # Define operations
#     predictions = last_rnn_output
    
    predictions = tf.squeeze(final_output,1) 
#     predictions = tf.nn.softmax(final_output, name='softmax_tensor')
#     predictions_final = tf.split(final_output,5,axis=0)
    
    if mode == Modes.PREDICT:
        prediction = {'Pred_RUL':final_output}
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(prediction)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=prediction, export_outputs=export_outputs)
    
    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.train.get_or_create_global_step()
        label_indices = tf.cast(labels, tf.float32)
        loss = tf.losses.mean_squared_error(label_indices, predictions)
        batch_size = tf.shape(labels)[0]
        total_loss = tf.to_float(batch_size) * loss
#         loss = tf.losses.softmax_cross_entropy(
#             onehot_labels=tf.one_hot(label_indices, depth=1), logits=logits)
#         tf.summary.scalar('OptimizeLoss', loss)

    if mode == Modes.TRAIN:
        print("LABEL==>")
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == Modes.EVAL:
        eval_metric_ops = {
            'rmse': tf.metrics.root_mean_squared_error(label_indices, predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)


def serving_input_fn(params):
    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [1,5,24], name = 'data')}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'feature_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    feature = tf.decode_raw(features['feature_raw'], tf.float64)
    feature.set_shape([120])
    feature = tf.cast(feature, tf.float32)
    label = tf.cast(features['label'], tf.int32)
    #print("===>", image.shape , label.shape, "<===")
    return feature, label


def train_input_fn(training_dir, params):
    return _input_fn(training_dir, 'train.tfrecords', batch_size=1)


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'test.tfrecords', batch_size=1)

def _input_fn(training_dir, training_filename, batch_size=1):
    test_file = os.path.join(training_dir, training_filename)
    filename_queue = tf.train.string_input_producer([test_file])
    
    feature, labels = read_and_decode(filename_queue)

    feature, labels = tf.train.batch(
        [feature, labels], batch_size=batch_size,
        capacity=10 + 3 * batch_size)

    return {INPUT_TENSOR_NAME: feature}, labels

'''

def _input_fn(tfrecords_path , training_filename , batch_size=1):

    dataset = (
    tf.data.TFRecordDataset(tfrecords_path)
    .map(parser)
    .batch(1)
    )
  
    iterator = dataset.make_one_shot_iterator()

    batch_feats, batch_labels = iterator.get_next()

    #return batch_feats, batch_labels
    return {INPUT_TENSOR_NAME: batch_feats}, batch_labels

'''



def neo_preprocess(payload, content_type):
    import logging
    import numpy as np
    import io
    logging.info('Invoking user-defined pre-processing function')

#     if content_type != 'text/csv' or content_type != 'application/vnd+python.numpy+binary':
#         raise RuntimeError('Content type must be application/x-image or application/vnd+python.numpy+binary')

#     f = payload
  
#     image = np.load(f)
    image={'data':payload}

    return image

### NOTE: this function cannot use MXNet
def neo_postprocess(result):
    import logging
    import numpy as np
    import json

    logging.info('Invoking user-defined post-processing function')

    # Softmax (assumes batch size 1)
    result = np.squeeze(result)
#     result_exp = np.exp(result - np.max(result))
#     result = result_exp / np.sum(result_exp)

    response_body = json.dumps(result.tolist())
    content_type = 'application/json'

    return response_body, content_type


