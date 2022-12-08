import gradio as gr
import wave
import matplotlib.pyplot as plt
import numpy as np
from extract_features import *
import pickle
import soundfile 
import librosa

classifier = pickle.load(open('finalized_rf.sav ', 'rb'))

def emotion_predict(input):
  input_features = extract_feature(input, mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True)
  rf_prediction = classifier.predict(input_features.reshape(1,-1))
  if rf_prediction == 'happy':
    return 'Happy 😎'
  elif rf_prediction == 'neutral':
    return 'Neutral 😐'
  elif rf_prediction == 'sad':
    return 'Sad 😢'
  else:
    return 'Angry 😤'
  

def plot_fig(input):
  wav = wave.open(input, 'r')

  raw = wav.readframes(-1)
  raw = np.frombuffer(raw, "int16")
  sampleRate = wav.getframerate()

  Time = np.linspace(0, len(raw)/sampleRate, num=len(raw))

  fig = plt.figure()

  plt.rcParams["figure.figsize"] = (50,15)

  plt.title("Waveform Of the Audio", fontsize=25)

  plt.xticks(fontsize=15)

  plt.yticks(fontsize=15)

  plt.ylabel("Amplitude", fontsize=25)

  plt.plot(Time, raw, color='red')

  return fig


with gr.Blocks() as app:
  gr.Markdown(
        """
    # Speech Emotion Detector 🎵😍
    This application classifies inputted audio 🔊 according to the verbal emotion into four categories:
    1. Happy 😎
    2. Neutral 😐
    3. Sad 😢
    4. Angry 😤
    """
  )
  with gr.Tab("Record Audio"):
    record_input = gr.Audio(source="microphone", type="filepath")
        
    with gr.Accordion("Audio Visualization", open=False):
      gr.Markdown(
          """
      ### Visualization will work only after Audio has been submitted
      """
      )    
      plot_record = gr.Button("Display Audio Signal")
      plot_record_c = gr.Plot(label='Waveform Of the Audio')
    
    record_button = gr.Button("Detect Emotion")
    record_output = gr.Text(label = 'Emotion Detected')

  with gr.Tab("Upload Audio File"):
    gr.Markdown(
        """
    ## Uploaded Audio should be of .wav format
    """
    )

    upload_input = gr.Audio(type="filepath")

    with gr.Accordion("Audio Visualization", open=False):
      gr.Markdown(
          """
      ### Visualization will work only after Audio has been submitted
      """
      )
      plot_upload = gr.Button("Display Audio Signal")
      plot_upload_c = gr.Plot(label='Waveform Of the Audio')

    upload_button = gr.Button("Detect Emotion")
    upload_output = gr.Text(label = 'Emotion Detected')
    
  record_button.click(emotion_predict, inputs=record_input, outputs=record_output)
  upload_button.click(emotion_predict, inputs=upload_input, outputs=upload_output)
  plot_record.click(plot_fig, inputs=record_input, outputs=plot_record_c)
  plot_upload.click(plot_fig, inputs=upload_input, outputs=plot_upload_c)

app.launch()