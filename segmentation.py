import pyaudio
import wave

FORMAT = pyaudio.paInt16  # Formato de áudio
CHANNELS = 2               # Número de canais (1 para mono, 2 para estéreo)
RATE = 44100               # Taxa de amostragem
CHUNK = 1024               # Tamanho do bloco de áudio
WAVE_OUTPUT_FILENAME = "gravacao.wav"  # Nome do arquivo de saída
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Gravando... Pressione Ctrl+C para parar.")

frames = []
try:
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
except KeyboardInterrupt:
    print("Gravação finalizada.")

stream.stop_stream()
stream.close()
audio.terminate()
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Gravacao salva como {WAVE_OUTPUT_FILENAME}")

import io, librosa
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy.signal import butter, lfilter
import soundfile as sf

audio_data, sample_rate = librosa.load('gravacao.wav', sr=16000)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

cutoff_freq = 2000.0
filtered_audio_data = butter_lowpass_filter(audio_data, cutoff_freq, sample_rate)
output_audio_path = 'segment.wav'
sf.write(output_audio_path, filtered_audio_data,sample_rate)

sound_file = AudioSegment.from_wav('segment.wav')
audio_chunks = split_on_silence(sound_file,
    min_silence_len=50,
    silence_thresh=-50
)

print(audio_chunks)

for i, chunk in enumerate(audio_chunks):
    out_file = "chunk{0}.wav".format(i)
    print(f'chunk{i}')
    chunk.export(out_file, format="wav")
