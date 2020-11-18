# These are the functions I need to process my audio. 
# They will apply to both the gender classifier and the language classifier, 
# so I want to be able to import them without rewriting them.

import librosa
SAMP_RATE = 16000
BATCH_SIZE = 32

print('Module initialized')
