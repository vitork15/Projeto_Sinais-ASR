import wave
import io, librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy.signal import butter, lfilter
import soundfile as sf
import keyboard
import pyaudio



FORMAT = pyaudio.paInt16  # Formato de áudio
CHANNELS = 2               # Número de canais (1 para mono, 2 para estéreo)
RATE = 8000              # Taxa de amostragem
CHUNK = 1024               # Tamanho do bloco de áudio
WAVE_OUTPUT_FILENAME = "segment.wav"  # Nome do arquivo de saída
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)


#start = True
#while(start):
#    if keyboard.is_pressed("space"):
#        start = False

print("Gravando... Pressione espaço para parar.")

start = True
frames = []
while start:
    data = stream.read(CHUNK)
    frames.append(data)
    if keyboard.is_pressed("space"):
        start = False

stream.stop_stream()
stream.close()
audio.terminate()
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Gravacao salva como {WAVE_OUTPUT_FILENAME}")

sound_file = AudioSegment.from_wav('/content/Projeto_Sinais-ASR/src/segment.wav')
audio_chunks = split_on_silence(sound_file,
    min_silence_len=150,
    silence_thresh=-50
)

print(audio_chunks)

for i, chunk in enumerate(audio_chunks):
    out_file = "chunk{0}.wav".format(i)
    print(f'chunk{i}')
    chunk.export(out_file, format="wav")
