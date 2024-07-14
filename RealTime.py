import tensorflow as tf
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

duration = 2
sample_rate = 24000  


model = tf.keras.models.load_model('./voice_model.tf', compile=False)

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

if __name__ == "__main__":
    real_time_classification()
