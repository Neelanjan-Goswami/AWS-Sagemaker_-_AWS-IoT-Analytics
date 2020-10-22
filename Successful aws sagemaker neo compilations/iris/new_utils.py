import tensorflow as tf
import numpy as np
import pandas as pd
import sys


import os

def convert_to_tfrecord(train,test):
    
    csv_root = './'
    #tfrecord_root = './dataset-tfrecord/'
    tfrecord_root = './'
    test_csv_file = test
    train_csv_file = train
    test_tfrecord_file = 'test.tfrecords'
    train_tfrecord_file = 'train.tfrecords'

    def _floatlist_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))

    def _int64list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # create the tfrecord dataset dir
    if not os.path.isdir(tfrecord_root):
        os.mkdir(tfrecord_root)

    for input_file, output_file in [(test_csv_file,test_tfrecord_file), (train_csv_file,train_tfrecord_file)]:
        # create the output file
        open(tfrecord_root + output_file, 'a').close()
        with tf.python_io.TFRecordWriter(tfrecord_root + output_file) as writer:
            with open(csv_root + input_file,'r') as f:
                f.readline() # skip first line
                for line in f:
                    feature = {
                        'sepal_length': _floatlist_feature(line.split(',')[0]),
                        'sepal_width': _floatlist_feature(line.split(',')[1]),
                        'petal_length': _floatlist_feature(line.split(',')[2]),
                        'petal_width': _floatlist_feature(line.split(',')[3]),
                    }
                    if f == train_csv_file:
                        feature['label'] = _int64list_feature(int(line.split(',')[4].rstrip()))
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())












'''
# 
# USAGE: $ python3 csv-to-tfrecords.py data.csv data.tfrecords
#


#infile=sys.argv[1]
#outfile=sys.argv[2]

csv = pandas.read_csv(infile, header=None).values


with tf.python_io.TFRecordWriter(outfile) as writer:
    for row in csv:
        
        ## READ FROM CSV ##
        
        # row is read as a single char string of the label and all my floats, so remove trailing whitespace and split
        row = row[0].rstrip().split(' ')
        # the first col is label, all rest are feats
        label = int(row[4])
        # convert each floating point feature from char to float to bytes
        feats = np.array([ float(feat) for feat in row[1:] ]).tostring()

        ## SAVE TO TFRECORDS ##

        # A tfrecords file is made up of tf.train.Example objects, and each of these
        # tf.train.Examples contains one or more "features"
        # use SequenceExample if you've got sequential data
        
        example = tf.train.Example()
        example.features.feature["feats"].bytes_list.value.append(feats)
        example.features.feature["label"].int64_list.value.append(label)
        writer.write(example.SerializeToString())
        
'''