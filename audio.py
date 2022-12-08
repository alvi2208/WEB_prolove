import speech_recognition as sr
from gtts import gTTS
import playsound as ps
from audioplayer import AudioPlayer

# Pengenalan Suara
recognizer = sr.Recognizer()
recognizer_exit = ['stop', 'bye', 'close']

# Process merekam suara
def voice_human():
    with sr.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic, 0.2)
        audio = recognizer.listen(mic)
        text = recognizer.recognize_google(audio, language='en')
        print(text.lower())
    return text.lower()

def detection(text):
    stt = gTTS(text=text, language='en')
    filename = 'dataset/cat-al.wav'
    stt.save(filename)
    try:
        AudioPlayer(filename).play(block=True)
        text = recognizer.recognize_google(stt, language='en')
    except:  
        print(text)