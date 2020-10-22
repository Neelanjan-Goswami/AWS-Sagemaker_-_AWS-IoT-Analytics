"""Converts MNIST data to TFRecords file format with Example protos."""
import os
import tensorflow as tf
import numpy as np
import pandas as pd



def data_preprocess(train_set, test_set):
    

    col = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Label']

    train_set = train_set.rename(columns={"120":"Sepal_Length", "4": "Sepal_Width", "setosa": "Petal_Length", "versicolor":"Petal_Width", "virginica": "Label" })
    test_set = test_set.rename(columns={"30":"Sepal_Length", "4": "Sepal_Width", "setosa": "Petal_Length", "versicolor":"Petal_Width", "virginica": "Label" })


    train_set_feat = train_set.drop(columns = ['Label'])
    train_set_label = train_set['Label']
    
    test_set_feat = test_set.drop(columns = ['Label'])
    test_set_label = test_set['Label']


    train_data_feat = np.array(train_set_feat)
    train_data_label = np.array(train_set_label)
    
    
    test_data_feat = np.array(test_set_feat) 
    test_data_label = np.array(test_set_label)
    
#     valid_data_feat = train_data_feat[:12]
#     valid_data_label = train_data_label[:12]
    
#     train_data_feat = train_data_feat[12:]
#     train_data_label = train_data_label[12:]
    
    

    
    data_set = {
    "train_dict" : {'features':train_data_feat , 'labels' : train_data_label },
    "test_dict" : {'features':test_data_feat , 'labels' : test_data_label }
    }
    return data_set



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def convert_to(dictdata, name, directory): #dictdata = data_set['train_dict']
    """Converts a dataset to tfrecords."""
    
    images = dictdata['features'] #ND arrays
    labels = dictdata['labels']  #ND arrays
    num_examples = labels.shape[0] #Number of examples

    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], num_examples))
    fshape = images.shape[1]


    filename = os.path.join(directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'fshape': _int64_feature(fshape),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
