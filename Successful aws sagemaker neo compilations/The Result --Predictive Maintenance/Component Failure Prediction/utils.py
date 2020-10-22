"""Converts MNIST data to TFRecords file format with Example protos."""
import os
import tensorflow as tf
import numpy as np
import pandas as pd



def data_preprocess_func(labeled_feat):
    

    test_results = []

    # split out training and test data
    train_Y = labeled_feat.loc[labeled_feat['datetime'] < pd.to_datetime('2015-07-31 01:00:00'), 'failure']
    print("train_y", train_Y)

    train_X = pd.get_dummies(labeled_feat.loc[labeled_feat['datetime'] < pd.to_datetime('2015-07-31 01:00:00')].drop(['failure', 'datetime'], 1))
    print("train_X", train_X)  
    
    
    
    cond = (labeled_feat['datetime'] > pd.to_datetime('2015-08-01 01:00:00')) & (labeled_feat['datetime']  < pd.to_datetime('2015-11-30 01:00:00'))
    
    print(cond)
    val_Y = labeled_feat.loc[cond , 'failure']
    
    print("val_Y", val_Y)
    
    
    val_X = pd.get_dummies(labeled_feat.loc[cond].drop(['failure', 'datetime'], 1))
    print("val_X", val_X) 
    
    
    
    
    
    test_Y = labeled_feat.loc[labeled_feat['datetime'] > pd.to_datetime('2015-12-01 01:00:00'), 'failure']
    print("test_Y", test_Y)
    test_X = pd.get_dummies(labeled_feat.loc[labeled_feat['datetime'] > pd.to_datetime('2015-12-01 01:00:00')].drop(['failure', 'datetime'], 1))
    print("test_X", test_X)
    
    



    train_set_feat = train_X
    train_set_label = train_Y
    
    test_set_feat = test_X
    test_set_label = test_Y


    train_data_feat = np.array(train_set_feat)
    train_data_label = np.array(train_set_label)
    
    
    test_data_feat = np.array(test_set_feat) 
    test_data_label = np.array(test_set_label)
    
    val_data_feat = np.array(val_X)
    val_data_label = np.array(val_Y)

    
    

    
    data_set = {
    "train_dict" : {'features':train_data_feat , 'labels' : train_data_label },
    "test_dict" : {'features':test_data_feat , 'labels' : test_data_label },
    "val_dict":{'features':val_data_feat , 'labels' : val_data_label}
        
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
    writer = tf.io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'fshape': _int64_feature(fshape),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
