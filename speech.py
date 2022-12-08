# -*- coding: utf-8 -*-
#import zipfile
#zip_ref = zipfile.ZipFile("dataset/train.zip.zip")
#zip_ref.extractall()
#zip_ref.close()

import os
import argparse 
import numpy as np
from scipy.io import wavfile 
from hmmlearn import hmm
import librosa
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile 
from librosa.feature import mfcc

class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, 
                    covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)

input_folder='aplikasi\dataset'

os.listdir(input_folder)

for dirname in os.listdir(input_folder):
  # Get the name of the subfolder 
  subfolder = os.path.join(input_folder, dirname)
  #print(subfolder)
  label = subfolder[subfolder.rfind('/') + 1:]
  print(label)

sampling_freq, audio = librosa.load("dataset\kata_benda\College_fazrin.wav")

mfcc_features = mfcc(sampling_freq,audio)

print('\nNumber of windows =', mfcc_features.shape[0])
print('Length of each feature =', mfcc_features.shape[1])

mfcc_features = mfcc_features.T
plt.matshow(mfcc_features)
plt.title('MFCC')

hmm_models = []
for dirname in os.listdir(input_folder):
    subfolder = os.path.join(input_folder, dirname)
    if not os.path.isdir(subfolder): 
         continue
    label = subfolder[subfolder.rfind('/') + 1:]
    X = np.array([])
    y_words = []
for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = librosa.load(filepath)            
            mfcc_features = mfcc(sampling_freq, audio)
            if len(X) == 0:
                X = mfcc_features[:,:15]
            else:
                X = np.append(X, mfcc_features[:,:15], axis=0)            
            y_words.append(label)
print('X.shape =', X.shape)
hmm_trainer = HMMTrainer()
hmm_trainer.train(X)
hmm_models.append((hmm_trainer, label))
hmm_trainer = None

# Test files
input_files = [
            'dataset\kata_benda\College_fazrin.wav',
            'dataset\kata_kerja\Don_t worry about it_ummu.wav',
            'dataset\kata_keterangan\OAF_goose_fear.wav',
            'dataset\kata_sifat\Difficult_fazrin.wav'
            ]

for input_file in input_files:
      sampling_freq, audio = librosa.load(input_file)

# Extract MFCC features
      mfcc_features = mfcc(sampling_freq, audio)
      mfcc_features=mfcc_features[:,:15]

      scores=[]
      for item in hmm_models:
          hmm_model, label = item
            
          score = hmm_model.get_score(mfcc_features)
          scores.append(score)
      index=np.array(scores).argmax()
      
      print("\nTrue:", input_file[input_file.find('/')+1:input_file.rfind('/')])
      print("Predicted:", hmm_models[index][1])

print(f'result : { hmm_models[index][1]}')

# Simpan model dalam bentuk format pkl (pickle)
with open('prolove.pkl', 'wb') as r:
  pickle.dump(hmm_models,r)