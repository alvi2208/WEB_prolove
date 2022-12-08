import pickle
import random
import string
import tensorflow as tf
from librosa.feature import mfcc
import pyttsx3

clf = pickle.load(open('prolove.pkl', 'rb'))

def make_prediction(input):
    input_features = extract_feature(input, mfcc=True, mel=True)
    predict = clf.predict(input_features.reshape(1,-1))
    if predict == 'kata_benda':
        return 'Kata Benda'
    elif predict == 'kata_kerja':
        return 'Kata Kerja'
    elif predict == 'kata_keterangan':
        return 'Kata Keterangan'
    elif predict == 'kata_sifat':
        return 'kata_sifat'
    else:
        return 'Cannot Prediction!'

