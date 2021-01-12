# AudioLanguageClassifer

# Introduction
The classification of audio files and acoustic data is an ongoing area of research in Machine Learning.The training of audio classifiers has been heavily influenced by advances in image classification (Hershey, et.al., 2017). Transfer learning has allowed well-trained image classifiers to be applied to audio data with fairly good success. For example, the VGGish audio classifier (see below) is based on the configuration of the VGG image classifier (Hershey, et.al., 2017, Simonyan & Zisserman, 2015). The current project seeks to employ transfer learning within the acoustic domain by using the VGGish model as the basis for classification of speech.  The goal is to see if a model trained for acoustic event detection (AED) can be leveraged to analyze speech data. Are the general acoustic features learned by the AED model sufficient to train a classifier for the narrower domain of human speech? Two classifier models will be developed, one to determine speaker gender and one to determine the native language of the speaker (limited to the top 10 languages present in the database, in addtion to English).

# VGGish
This project uses the VGGish model as an acoustic feature extractor. VGGish accepts as input audio files sampled at 16kHz, converts the files to mel spectrograms, and outputs an array of 128-dimensional features. In this project, the 128-dimensional features are then fed into a trainable classifier model. The pre-trained VGGish model is available on 
[GitHub](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) or through the [Tensorflow Hub](https://tfhub.dev/google/vggish/1).

# The Data
The audio speech data came from the Speech Accent Archive, collected by Weinberger (2015) and available for download from [Kaggle](https://www.kaggle.com/rtatman/speech-accent-archive). The data includes demographic information about 2172 speakers, as well as 2138 audio recordings of speakers reading a fixed passage in English. The demographic information records 9 features, including speaker ID, native language, birthplace, country, sex, and filename.

## Exploratory Data Analysis
After processing, the final dataset included roughly equal numbers of male and female speakers, as shown here.

![Distribution of Gender across all speakers.](/figures/GenderDistAll.png)

After processing, the data contained speakers from 199 native languages, 78 of which were represented by a single speakers. English was the native language most represented, with 579 speakers. After English, the ten most represented languages were selected for use in the Language Classifier. The number of speakers from these languages varied from 36 to 162, as shown in Figure 3.

![Count of speakers in the 10 languages with most speakers in the dataset (excluding English).](/figures/TopLangCount.png)

## Audio file processing

The VGGish model accepts audio files as input and output arrays of feature embeddings. Before being fed to the VGGish model, the audio files from the database were normalized, divided into 10s segments, and augmented with random noise.

Given the length of the reading passage, the audio files for most speakers were longer than 20s. Dividing the files into 10s segments accomplished several things. First, it supplied audio files for the VGGish models that were the same length as the VGGish training files. Second, it insured that the dimensions of the VGGish output would be consistent, since all of the input files would be of the same length. Finally, it increased the number and variability of files available for the training and testing of the new classifier models, which should lead to more robust models.

Since the original audio files differed in length between speakers, the number of segments per speakers was not consistent. As shown below, the audio file from a native English speaker was just over 20 seconds long, as was divided into two segments. The file from a native Arabic speaker, by contrast, was over 40 seconds long, and was divided into four segments. Each segment was saved as a separate .wav file.

![Waveforms of audio for speakers English11 and Arabic24. Dashed red lines show 10s segments that will be saved upon segmentation.](/figures/Waveforms.png)

Once the original audio files had been segmented and saved, the segments were augmented by adding randomly generated, low-level 'noise', as shown below. The addition of noise should improve models robustness, since the model will learn to focus on salient acoustic features, while ignoring the noise. The noisy segments and the original segments were saved in separate files, thus doubling the number of files available for model training.

![Waveforms of original (left) and noise augmented (right) segments from speaker English11. The low-level, random noise is most noticeable when comparing portions where the original waveform has little sound (e.g. 0-0.25 s, 4.25-4.75 s).](/figures/OrigNoise.png)

Internally, the VGGish model converts the audio files to a Mel spectrogram, which is then run through the convolutional layers to produce the feature embedding. An example of a Mel spectrogram is provided in the right panel of the figure below.

![Waveform (left) and Mel spectrogram (right) of a segment from speaker English11. The VGGish model converts audio input files to a Mel spectrogram, which is then fed through its convolutional layers.](/figures/English11MelSpec.png)

# Gender Classifier
The goal of the Gender Classifier is to predict speaker gender (female and male, coded as 0 and 1, respectively) from the audio input. All of the speakers from the Speech Accent Archive were used for the Gender Classifier. First, the speakers were split into training, validation and testing sets.  Next, the audio files for each dataset were segmented. Then, the segments in the training and validation datasets were augmented with noise. Since there was not a consistent number of segments per speaker, and since the testing file were not augmented with noise, the distribution of speakers and segments in the training and testing sets differed. Most notably, while there were fewer speakers in the validation set than the testing set, once the files were segmented and augmented, there were more segments in the validation set. However, the distribution of male and female speakers/segments remained fairly consistent among the datasets.
## Model Architecture
Two different models were trained and evaluated for the gender classifier. The models had identical input and output layers, and differed in the number of hidden layers. Each hidden layer consisted of a dense layer followed by a dropout layer with a dropout rate of 50\%. The first model contained a single hidden layer with 128 nodes. The second model contained two hidden layers, the first with 128 nodes, and the second with 64 nodes. Prior to being fed into the output layer, the output from the previous layer was flattened into a 1D array. The shape of the flattened array varied for each model, based on the number of nodes of the prior layer (128 vs 64). Since the classification was binary, the output layer contained one node, with a sigmoid activation function.
## Results
The single-layer model had overall accuracy of 98.14% and showed similar performance between both gender classes, with nearly equal numbers of speakers being misclassified from each class. The two-layer model had overall accuracy of 97.55%, but showed more discrepancies between the performance of the two classes. Male voices were identified correctly at a rate of 99.14%, while female voices were identified correctly only 96.5%. However, the precision of the predictions was the reverse - the precision of female voice prediction was 99.14\%, while the precision of male voice prediction was 96.02\%.

# Language Classifier
The goal of the Language Classifier was to identify the native language of the speaker, given that all speakers read the same passage in English. To limit the scope of the classifier, the ten most represented languages in addition to English were selected for use in the Language Classifier, as shown above. The labels for the classifier were one-hot encoded into a 1x11 array.

The selected data contained many more speakers of English, Spanish and Arabic than speakers of other languages. To balance the classes for the Language Classifier, the number of speakers for these three languages was downsampled to 75, which meant that each of these language classes contained 12.0773% of the total number of speakers in the dataset.  After the speakers were split into training, validation and testing sets, the audio files for all speakers were segmented, and the segments in the training and validation splits were augmented with noise. Since the number of segments per speaker was dependent on the length of the original audio file, which was not determined before the speakers were split, the number of segments per language and per data split were not identical to the distribution of the speakers. Like with the Gender Classifier data, the validation data contained fewer speakers, but more segments than the test data.

While the distribution of speakers and segments for most of the languages remained similar, the distribution of Arabic and English segments changed dramatically. While there were equal numbers of Arabic and English speakers, there were 380 Arabic segments compared to 239 English segments. Thus, the distribution of Arabic segments grew to 15.109344% of the segments, while English segments shrank to 9.502982%, moving English behind Spanish, Mandarin and French.

## Model Architecture
Three models were trained and evaluated for the language classifier. All of the models had identical input and output layers. The models differed in the number of nodes per layer and the number of hidden layers. Each hidden layer consisted of a dense layer followed by a dropout layer with a dropout rate of 50%. The first model contained a single hidden layer with 12 nodes, while the second model had a single layer with 128 nodes. The third model contained two hidden layers, the first with 128 nodes, and the second with 64 nodes. Prior to being fed into the output layer, the output from the previous layer was flattened into a 1D array. The shape of the flattened array varied for each model, based on the number of nodes of the prior layer (12 vs 128 vs 64). There were 11 possible classes, so the output layer consisted of 11 nodes. A softmax activation function was used on the output layer, so that the output vector contained the probability that the speaker belonged to each of the output classes. The class with the highest probability was taken to the be predicted class.

## Results
A naive classifier that always predicted the majority class (Arabic) would have an accuracy of 15\%.  All three of the models improved upon this baseline accuracy rate, with accuracy rates of 23.2955%, 23.8636% and 0.25%. The single-layer, 12-node model had the highest precision rate, but has the lowest accuracy, recall and F1 scores. The single-layer, 128-node model had the highest recall and F1 scores, but also the lowest precision. The two-layer model had the highest overall accuracy, and has F1 scores only slightly lower than the single-layer, 128-node model. 

Given the slightly higher rates of precision and accuracy, the two-layer model was chosen as the final model.

## Future directions
There are several ways that future Language Classifier models could be optimized further. From a model architecture standpoint, additional models could be developed that included more layers and/or more nodes, had different dropout proportions, or used different activation functions. From a phonetic/acoustic standpoint, there may be additional acoustic features, like tempo/beat tracking, or the duration of the original file, that could be added to improve discrimination between phonetically different accents. Another possible feature to include in the model could be an accentedness rating - how native-like or non-native-like the speech sounds to other speakers. While this rating would not necessarily be available for future samples (in production), it could be useful for training.

While imbalanced data is prevalent in the real world, it is possible that using a more balanced data set, either by downsampling the existing data, or by collecting more speech samples from underrepresented languages, would increase model performance. Finally, a model trained to distinguish between fewer languages would have better accuracy, if the final use case involved processing audio files from only a subset of the language classes.

# Conclusions
The goal of this project was to use a model pre-trained for acoustic event detection and evaluate if the extracted features were useful for analyzing speech. In both cases, the features extracted by the VGGish model were useful for training classifiers to identify speaker characteristics from audio files. The VGGish embeddings were sufficient to train a Gender Classifier with 98\% accuracy. The Language Classifier based on the VGGish features showed improvement over a model that only predicted the majority class. Transfer learning has been shown to be useful for training new models, even with slightly different domains, with smaller datasets in relatively short amounts of time.
