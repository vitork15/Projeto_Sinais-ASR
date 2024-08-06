import librosa
from hmmlearn import hmm
import numpy as np
import scipy
import pickle
import os

dirname = os.path.dirname(__file__) + '/'

def pre_process(audio, sr, cutoff_freq):
    # y[n] = x[n]-0.97*x[n-1]
    res = librosa.effects.preemphasis(audio, coef = 0.97)

    # aplicando o filtro passa-baixa em cutoff_frequency
    sos = scipy.signal.butter(N=5, Wn=cutoff_freq, btype='lowpass', fs=sr, output = 'sos')
    res = scipy.signal.sosfiltfilt(sos, res)

    return res

def training():

    for number in range(10):
        mfcc_vector = np.array([]).reshape(0,20)
        lengths = []
        for speaker in range(1,48):
            for iteration in range(5):
                print(f"Gerando MFCCs do falante {speaker}, número {number}, iteração {iteration}")
                if speaker < 10:
                    filename = dirname + 'data/0' + str(speaker) + '/' + str(number) + '_0' + str(speaker) + '_' + str(iteration) + '.wav'
                else:
                    filename = dirname + 'data/' + str(speaker) + '/' + str(number) + '_' + str(speaker) + '_' + str(iteration) + '.wav'
                audio, sr = librosa.load(filename, sr = 8000)
                audio = pre_process(audio, sr, 3400)
                mfccs = librosa.feature.mfcc(y=audio, sr=sr)
                mfcc_vector = np.concatenate((mfcc_vector,mfccs.transpose()))
                lengths.append(len(mfccs.transpose()))

        model = hmm.GaussianHMM(n_components=5, init_params='st', covariance_type="full", n_iter = 100)


        print(f"Iniciando treinamento do modelo para o número {number}")
        model.fit(mfcc_vector, lengths)

        for i in range(4):
            print(f"Reinicializando modelo: iteração {i+1}")
            modelgreedy = hmm.GaussianHMM(n_components=5, init_params='st', covariance_type="full", n_iter = 100)
            modelgreedy.fit(mfcc_vector, lengths)
            if(modelgreedy.score(mfcc_vector, lengths) > model.score(mfcc_vector, lengths)): 
                model = modelgreedy
                print(f"Achou melhora na iteração {i+1}")


        modelname = 'model' + str(number)
        with open(dirname + modelname + '.pkl', "wb") as file: pickle.dump(model, file)

def testing():

    model = []

    for testnumber in range(10):
        with open(dirname + 'model' + str(testnumber) + '.pkl', "rb") as file: 
            model.append(pickle.load(file))
            file.close()

    counter = 0
    total = 10*12*5

    for number in range(10):
        for speaker in range(48,60):
            for iteration in range(5):
                #print(f"Gerando MFCCs do falante {speaker}, número {number}, iteração {iteration}")
                if speaker < 10:
                    filename = dirname + 'data/0' + str(speaker) + '/' + str(number) + '_0' + str(speaker) + '_' + str(iteration) + '.wav'
                else:
                    filename = dirname + 'data/' + str(speaker) + '/' + str(number) + '_' + str(speaker) + '_' + str(iteration) + '.wav'
                audio, sr = librosa.load(filename, sr = 8000)
                audio = pre_process(audio, sr, 3400)
                mfccs = librosa.feature.mfcc(y=audio, sr=sr)
                result = -1
                maxscore = -9999999.0
                for testnumber in range(3):
                    score = model[testnumber].score(mfccs.transpose())
                    if(maxscore < score):
                        maxscore = score
                        result = testnumber
                if result == number: counter = counter + 1

    print(f"Acurácia: {counter/total}")

def get_number(filename):

    model = []

    for testnumber in range(10):
        with open(dirname + 'model' + str(testnumber) + '.pkl', "rb") as file: 
            model.append(pickle.load(file))
            file.close()

    audio, sr = librosa.load(filename, sr = 8000)
    audio = pre_process(audio, sr, 3400)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr)
    result = -1
    maxscore = -9999999.0
    for testnumber in range(10):
        score = model[testnumber].score(mfccs.transpose())
        if(maxscore < score):
            maxscore = score
            result = testnumber
    
    return result


def main():
    training()
    testing()


if __name__ == "__main__":
    main()