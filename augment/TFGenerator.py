# Create a tensorflow dataset generator
# Adapted from https://biswajitsahoo1111.github.io/post/efficiently-reading-multiple-files-in-tensorflow-2/
import numpy as np
import librosa
import re
import tensorflow as tf
import tensorflow_hub as hub

from keras.layers import Dense
from keras.models import Sequential

# Load VGGish model
# Link to the model on TFHub
hub_url = 'https://tfhub.dev/google/vggish/1'

# Load the model as a Keras model
vggish_model = hub.KerasLayer(hub_url)
vggish_model.trainable = False

def tf_data_generator(file_list, batch_size=32):
    """ Create a dataset generator. 
    Iterate through a list of filenames and process in batches.
    Extract audio features from vggish model.
    WARNING: This generator forms an infinite loop, 
    so you need to specify how long to run the generator 
    before fitting and evaluating a model.

    Arguments:
    file_list - list of filenames to iterate
    vggish_model  - pass the instantiated model to the function
    batch_size - how many files to process at a time
    """
    i = 0
    while True: #infinite loop
        if i*batch_size >= len(file_list):
            i=0
            np.random.shuffle(file_list)
        else:
            file_chunk = file_list[i*batch_size:(i+1)*batch_size]
            data = []
            labels = []
            label_classes = tf.constant(['M', 'F'])
            for file in file_chunk:
                # Read data
                audio, sr = librosa.load(file, sr=16000)
                # Apply transformations
                embed = vggish_model(audio)
                data.append(embed)
                # Extract labels from filename
                bytes_string = file
                string_name = str(bytes_string, 'utf-8')
                split_str = string_name.split('.')
                pattern = tf.constant(split_str[2])
                for j in range(len(label_classes)):
                    if re.match(pattern.numpy(), label_classes[j].numpy()):
                        labels.append(j)

            data = np.asarray(data)
            labels = np.asarray(labels)

            yield data, labels
            i += 1