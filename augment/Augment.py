# These are the functions I need to process my audio. 
# They will apply to both the gender classifier and the language classifier, 
# so I want to be able to import them without rewriting them.
import librosa
import soundfile as sf
import numpy as np

def normalize(audio):
    """Normalize audio file to range [-1, 1]"""
    norm = audio/max(audio)
    return norm

def segment_10s(audio, sr):
    """ Load an audio file and divide into 10 second segments. 
    Arguments:
    audio - the audio file
    sr - sampling rate of the audio file
    Returns: a dictionary of audio segments. 
    Key is the segment index, value is the segmented audio.
    """
    seg_files = {}
    n_seg = int((len(audio)/sr)/10)
    for i in range(n_seg):
        segment = audio[10*i*sr:(i+1)*10*sr]
        seg_files[i] = segment
    return seg_files

def segment_audio(filename, y_value, split='train', clf='gender'):
    """Load an audio file and segment into 10s increments
    Save each segment to the target directory.
    Append the gender of the speaker and the segment index to the filename.

    Arguments:
    filename - base name of audio file (without .mp3 extension)
    y_value - class label
    split - 'train' or 'test' data, for filepath
    clf - 'gender' or 'lang10' for filepath
    """

    filepath = 'recordings/recordings/' + filename + '.mp3'
    audio, sr = librosa.load(filepath, sr=16000)
    audio = normalize(audio)

    # Add gender label to filename for later processing
    sex = y_value
    if sex == 'female':
        filename = '{}.F'.format(filename)
    else: filename = '{}.M'.format(filename)

    # Segment audio file
    seg_files = segment_10s(audio, sr)

    for key, val in seg_files.items():
        new_name = '{}.{}'.format(filename, key)
        sf.write('data/{}/{}/{}o.wav'.format(clf, split, new_name), val, sr)

def add_noise(audio):
    '''
    Add random noise to an audio file.
    Arguments: audio - the audio file
    Returns: the noisy audio file
    ''' 
    # Load random number generator
    rng = np.random.default_rng()
    # Generate random noise
    noise = rng.standard_normal(len(audio))
    # Add noise to file
    noisy_seg = audio + 0.005*noise

    return noisy_seg

def noisy_data(filename, split='train', clf='gender'):
    """Load an audio file (or segment).   
    Add random noise to the file and save with new filename.

    Arguments:
    filename - filename/segment base, without 'o.wav'
    split - 'train' or 'test' data, for filepath
    clf - 'gender' or 'lang10' for filepath
    """

    filepath = 'data/{}/{}/{}o.wav'.format(clf, split, filename)
    audio, sr = librosa.load(filepath, sr=16000)
 
    # Add noise
    noisy = add_noise(audio)
    # Write noise to file
    sf.write('data/{}/{}/{}n.wav'.format(clf, split, filename), noisy, sr)
    #print("Noise added to {}".format(filename))

print("Augment scripts reloaded")