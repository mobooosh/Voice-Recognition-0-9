import tensorflow as tf
import numpy as np
import librosa
import sys
import os
import sounddevice as sd
from platform import python_version
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from scipy.io.wavfile import write




model = tf.keras.models.load_model('./voice_model.tf', compile=False)

def rec_online():
    
    duration = 2
    sample_rate = 24000  
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def get_spectrogram(audio):
        frame_length = 320
        frame_step = 45
        spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, -1)
        return spectrogram
    
    def record_audio(duration, sample_rate):
        print("Say the number:")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio = np.squeeze(audio)
        return audio
    
    def classify_audio(audio):
        if len(audio) < sample_rate * duration:
            audio = np.pad(audio, (0, sample_rate * duration - len(audio)))
        elif len(audio) > sample_rate * duration:
            audio = audio[:sample_rate * duration]
    
        spectrogram = get_spectrogram(audio)
    
    
        spectrogram = tf.expand_dims(spectrogram, 0)
    
        predictions = model.predict(spectrogram)
        predicted_label = np.argmax(predictions, axis=1)
        return predicted_label[0]
    
    def real_time_classification():
        while True:
            audio = record_audio(2, 24000)
            label = classify_audio(audio)
            print(f'Predicted number: {label}')

    return real_time_classification()

def rec_ofline(file_path):
    padd = 48000

    def load_wav_16k_mono(path):
        file_contents = tf.io.read_file(path)
        wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        return wav
    
    def predict(file_path):
        wav = load_wav_16k_mono(file_path)
        wav = wav[:padd]
        zero_padding = tf.zeros([padd] - tf.shape(wav), dtype=tf.float32)
        wav = tf.concat([zero_padding, wav], 0)
        spectogram = tf.signal.stft(wav, frame_length=320, frame_step=45)
        spectogram = tf.abs(spectogram)
        spectogram = tf.expand_dims(spectogram, axis=2)
        spectogram = tf.expand_dims(spectogram, axis=0)
        prediction = model(spectogram)
        return prediction.numpy().argmax()
        
    return(predict(file_path))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
        print(rec_ofline(file_path))
    else:
        rec_online()
