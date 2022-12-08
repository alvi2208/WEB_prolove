import json
import random
import nltk
import string
import numpy as np
import pickle
import tensorflow as tf
from process import preparation, generate_response
from flask import Flask, render_template, request
from audio import *

# download nltk
preparation()

#Sflask
app = Flask(__name__)

#get audio from drive 
demo_mfcc, demo_pitch, demo_mag, demo_chrom = get_audio_features(demo_audio_path, sampling_rate)
mfcc = pd.Series(demo_mfcc)
pit = pd.Series(demo_pitch)
mag = pd.Series(demo_mag)
C = pd.Series(demo_chrom)

demo_audio_features= np.expand_dims(demo_audio_features, axis=0)
demo_audio_features= np.expand_dims(demo_audio_features, axis=2)
demo_audio_features.shape
demo_preds = (demo_audio_features)

#load model
loaded_model.predict(demo_audio_features, batch_size=32, verbose=1)
demo_preds
index = demo_preds.argmax(axis=1).item()
index

#start
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)
    return result

@app.route("/record")
def record():
    text = dengerin()
    # result = generate_response(text)
    # bilang(text)
    return text

@app.route("/speak")
def speak():
    user_input = str(request.args.get('msg'))
    bilang(user_input)

if __name__ == "__main__":
    app.run(debug=True)