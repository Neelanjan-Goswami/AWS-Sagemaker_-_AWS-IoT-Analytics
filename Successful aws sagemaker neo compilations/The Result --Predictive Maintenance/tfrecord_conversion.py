import os
import tensorflow as tf
import numpy as np
import pandas as pd

def lstm_preprocess(df,timesteps):
    df = df.drop(columns = ['id', 'cycle', 'RUL'])
    features = df.columns
    
    df_list = [df[features].shift(shift_val) if (shift_val == 0) 
                                else df[features].shift(-shift_val).add_suffix(f'_{shift_val}') 
                                for shift_val in range(0,timesteps)]
    
    df_concat = pd.concat(df_list, axis=1, sort=False)
    
    df_concat = df_concat.iloc[:-timesteps,:]
    X = pd.DataFrame(df_concat)
    
    X_trans = np.empty((X.shape[0], timesteps, 0))

    # Adjusting the shape of the data
    for feat in features:
        # Regular expressions to filter each feature and
        # drop the NaN values generated from the shift
        df_filtered = X.filter(regex=f'{feat}(_|$)')
        df_filtered = df_filtered.values.reshape(df_filtered.shape[0], timesteps, 1)
        X_trans = np.append(X_trans, df_filtered, axis=2)
#         print(feat, df_filtered.shape)
    
    return(X_trans)


def data_preprocess(train_set, test_set):

#     train_set_feat = train_set.drop(columns = ['RUL'])
    train_set_feat = lstm_preprocess(train_set, 5)
    train_set_label = train_set['RUL']
    
#     test_set_feat = test_set.drop(columns = ['RUL'])
    test_set_feat = lstm_preprocess(test_set, 5)
    test_set_label = test_set['RUL']


    train_data_feat = np.array(train_set_feat)
    train_data_label = np.array(train_set_label)[:len(train_data_feat)]
    
    
    test_data_feat = np.array(test_set_feat) 
    test_data_label = np.array(test_set_label)[:len(test_data_feat)]
    
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
    
    features = dictdata['features'] #ND arrays
    labels = dictdata['labels']  #ND arrays
    num_examples = labels.shape[0] #Number of examples

    if features.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (features.shape[0], num_examples))
    timestep = features.shape[1]
    fshape = features.shape[2]
    


    filename = os.path.join(directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        feature_raw = features[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'timestep': _int64_feature(timestep),
            'fshape': _int64_feature(fshape),
            'label': _int64_feature(int(labels[index])),
            'feature_raw': _bytes_feature(feature_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
