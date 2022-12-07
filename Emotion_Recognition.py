#!/usr/bin/env python
# coding: utf-8

# In[38]:


import soundfile as sf # to read audio file
import numpy as np
import librosa as lb # to read audio file
import glob
import os
from sklearn.model_selection import train_test_split as tts # for splitting training and testing
from sklearn.neural_network import MLPClassifier as mlp # multi-layer perceptron model
from sklearn.metrics import accuracy_score as acc # to measure how good the accuracy is
from sklearn.metrics import classification_report as cr 
import pickle as pk # to save model after training


# In[39]:


# all emotions on the dataset
emotions_present = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# we allow only these emotions
emotions_allowed = {
    "angry",
    "sad",
    "neutral",
    "happy"
}


# In[40]:



def get_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = get_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(lb.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(lb.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(lb.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(lb.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(lb.feature.tonnetz(y=lb.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result


# In[52]:


def loading_data(test_size=0.2):
    X, y = [], []
    for file in glob.glob("E:\\Audio\\Actor_*\\*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # get the emotion label
        emotion = emotions_present[basename.split("-")[2]]
        # we allow only emotions_allowed we set
        if emotion not in emotions_allowed:
            continue
        # extract speech features
        features = get_feature(file, mfcc=True, chroma=True, mel=True)
        # add to data
        X.append(features)
        y.append(emotion)
    # split the data to training and testing and return it
    return tts(np.array(X), y, test_size=test_size,train_size=0.75, random_state=123)


# In[53]:


# load the dataset, 75% training & 25% testing
X_train, X_test, y_train, y_test = loading_data(test_size=0.25)


# In[54]:


# print some details
# number of samples in training data
print("[+] Number of training samples:", X_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape[0])
# number of features used
# this is a vector of features extracted
# using utils.get_features() method
print("[+] Number of features:", X_train.shape[1])


# In[55]:


# best model, determined by a grid search
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 500,
}


# In[56]:



# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
model = mlp(**model_params,random_state=900)


# In[57]:


# train the model
print("[*] Training the model...")
model.fit(X_train, y_train)


# In[58]:


# predict 25% of data to measure how good it is
y_pred = model.predict(X_test)


# In[59]:


# calculate the accuracy with 75% training data and 25% testing data
accuracy = acc(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))


# In[51]:


# Creating the classification report
print(cr(y_test,y_pred))


# In[ ]:




