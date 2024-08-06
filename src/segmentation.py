import wave
import io, librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy.signal import butter, lfilter
import soundfile as sf
import keyboard

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
