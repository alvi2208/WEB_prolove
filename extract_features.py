import numpy as np
import soundfile
import librosa

def extract_feature(file_name, **kwargs):
    
    chroma = kwargs.get("chroma")
    contrast = kwargs.get("contrast")
    mfcc = kwargs.get("mfcc")
    mel = kwargs.get("mel")
    tonnetz = kwargs.get("tonnetz")
    
    with soundfile.SoundFile(file_name) as audio_clip:
        X = audio_clip.read(dtype="float32")
        sound_fourier = np.abs(librosa.stft(X))   # Conducting short time fourier transform of audio clip
        result = np.array([])
        
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=audio_clip.samplerate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=sound_fourier, sr=audio_clip.samplerate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=audio_clip.samplerate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=sound_fourier, sr=audio_clip.samplerate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=audio_clip.samplerate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result