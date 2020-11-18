# These are the functions I need to process my audio. 
# They will apply to both the gender classifier and the language classifier, 
# so I want to be able to import them without rewriting them.

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

def segment_data(x_names, y_values, split='train', clf='gender'):
    """Load a list of original file names. 
    Iterate through the list and load each audio file. 
    Segment each file into 10s pieces and 
    save each segment to the target directory.

    Arguments:
    x_names - pd Series of filenames
    y_values - pd Series class labels
    split - 'train' or 'test' data, for filepath
    clf - 'gender' or 'lang10' for filepath
    """
    seg_names = []

    for i in range(len(x_names)):
        filename = x_names.iloc[i]
        filepath = 'recordings/recordings/' + filename + '.mp3'
        audio, sr = librosa.load(filepath, sr=16000)
        audio = normalize(audio)

        # Add gender label to filename for later processing
        sex = y_names.iloc[i]
        if sex == 'female':
            filename = '{}.F'.format(filename)
        else: filename = '{}.M'.format)filename)

        # Segment audio file
        seg_files = segment_10s(audio, sr)
        for key, val in seg_files.items():
            new_name = '{}.{}'.format(filename, key)
            sf.write('data/{}/{}/{}o.wav'.format(clf, split, new_name), val, sr)
            seg_names.append(new_name)
    return seg_names

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

def noisy_data(x_names, split='train', clf='gender'):
    """Load a list of segment names. 
    Iterate through the list and load each audio file. 
    Add random noise to each file and save with new filename.

    Arguments:
    x_names - pd Series of filenames
    split - 'train' or 'test' data, for filepath
    clf - 'gender' or 'lang10' for filepath
    """
  for i in range(len(x_seg_list)): #list of seg_names
    filename = x_seg_list[i]
    filepath = 'data/{}/{}/{}o.wav'.format(clf, split, filename)
    audio, sr = librosa.load(filepath, sr=16000)
 
    # Add noise
    noisy = add_noise(audio)
    # Write noise to file
    sf.write('data/{}/{}/{}n.wav'.format(clf, split, filename), noisy, sr)
    #print("Noise added to {}".format(x_names[i]))