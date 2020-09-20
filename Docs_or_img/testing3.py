# import libraries
import time
import os
from PIL import Image
from dlr import DLRModel
import numpy as np
import utils
import logging

logging.basicConfig(filename='test-dlr.log', level=logging.DEBUG)

current_milli_time = lambda: int(round(time.time() * 1000))


def run_inference():
   
    os.system('fswebcam -r 1024x768 --no-banner --scale 224x224 output.jpg -S 7 --save /home/pi/Photos/std.jpg') # uses Fswebcam to take picture
    image = Image.open('output.jpg')
    #data = np.array(image,dtype='float64')
    #data=data1.reshape((1,data1.shape[2],data1.shape[0],data1.shape[1]))

    #np.save( 'flamingo.npy', data)
 
    image_data = utils.transform_image(image)
    print(image_data)
    flattened_data = image_data.astype(np.float32).flatten()
    #np.save( 'puppi.npy',flattened_data)
    print("Start Prinring Flattern") 
    print(flattened_data)
    #run_inference(image_data)
    #time.sleep(15) # this line creates a 15 second delay before repeating the loop
   

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../model-rasp3b')
    batch_size = 1
    channels = 3
    height = width = 224
    input_shape = {'input0': [batch_size, channels, height, width]}
    classes = 1000
    output_shape = [batch_size, classes]
    device = 'cpu'
    model = DLRModel(model_path, input_shape, output_shape, device)

    synset_path = os.path.join(model_path, 'imagenet1000_clsidx_to_labels.txt')
    with open(synset_path, 'r') as f:
        synset = eval(f.read())

    #image = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dog.npy')).astype(np.float32)
    #input_data = {'data': image_data}
 
    # Predict
    out = model.run({'input0' : flattened_data}).squeeze()
    top1 = np.argmax(out)
    prob = np.max(out)
    print("Class: %s, probability: %f" % (synset[top1], prob))


    for rep in range(4):
        t1 = current_milli_time()
        out = model.run({'input0' : flattened_data}).squeeze()
        t2 = current_milli_time()

        logging.debug('done m.run(), time (ms): {}'.format(t2 - t1))

        top1 = np.argmax(out[0])
        logging.debug('Inference result: {}, {}'.format(top1, synset[top1]))
    
    import resource
    logging.debug("peak memory usage (bytes on OS X, kilobytes on Linux) {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    return {
        'synset_id': top1,
        'prediction': synset[top1],
        'time': t2 - t1
    }

if __name__ == '__main__':
    res = run_inference()
    cls_id = res['synset_id'] 
    print("All tests PASSED!")





