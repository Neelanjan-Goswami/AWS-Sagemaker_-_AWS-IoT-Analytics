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

    input_layer = tf.reshape(features[INPUT_TENSOR_NAME], [-1,4])




    # Dense Layer

    dense1 = tf.layers.dense(inputs=input_layer, units=10, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=20, activation=tf.nn.relu)
    dense3 = tf.layers.dense(inputs=dense2, units=10, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense3, units=3, activation=None)
    

    # Define operations
    if mode in (Modes.PREDICT, Modes.EVAL):
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.train.get_or_create_global_step()
        label_indices = tf.cast(labels, tf.int32)
        
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(label_indices, depth=3), logits=logits)
        tf.summary.scalar('OptimizeLoss', loss)

    if mode == Modes.PREDICT:
        predictions = {
            'probabilities': probabilities
        }
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    if mode == Modes.TRAIN:
        print("LABEL==>")
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == Modes.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(label_indices, tf.argmax(input=logits, axis=1))
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)

'''


def estimator_fn(run_config, params):
    feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[4])]

    return tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                      hidden_units=[10, 20, 10],
                                      n_classes=3,
                                      config=run_config)


def serving_input_fn():
    feature_spec = tf.FixedLenFeature(dtype=tf.float32, shape=[4])

    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()
'''



def serving_input_fn(params):
    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [1,4], name = 'data')}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

'''

def parser(record):

    features={
    'feats': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64)
     }
   
    
    parsed = tf.parse_single_example(record, features)
    #feats = tf.convert_to_tensor(parsed['feats'])
    #feats = tf.decode_raw(parsed['feats'], tf.float32)
    #feats.set_shape([4])
    #image = tf.cast(feats, tf.float32)
    feats = tf.convert_to_tensor(tf.decode_raw(parsed['feats'], tf.float64))
    label = tf.cast(parsed['label'], tf.int32)

    return {'feats': feats}, label


'''
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], tf.float64)
    image.set_shape([4])
    image = tf.cast(image, tf.float32)
    label = tf.cast(features['label'], tf.int32)
    #print("===>", image.shape , label.shape, "<===")
    return image, label


def train_input_fn(training_dir, params):
    return _input_fn(training_dir, 'train.tfrecords', batch_size=1)


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'test.tfrecords', batch_size=1)

def _input_fn(training_dir, training_filename, batch_size=1):
    test_file = os.path.join(training_dir, training_filename)
    filename_queue = tf.train.string_input_producer([test_file])
    
    image, label = read_and_decode(filename_queue)

    images, labels = tf.train.batch(
        [image, label], batch_size=batch_size,
        capacity=10 + 3 * batch_size)

    return {INPUT_TENSOR_NAME: images}, labels

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

    if content_type != 'application/x-image' and content_type != 'application/vnd+python.numpy+binary':
        raise RuntimeError('Content type must be application/x-image or application/vnd+python.numpy+binary')

    f = io.BytesIO(payload)
  
    image = np.load(f)

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


