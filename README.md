# Projeto_Sinais-ASR
Projeto da cadeira de Sinais e Sistemas para o desenvolvimento de um sistema de reconhecimento automático de fala utilizando Modelos Ocultos de Markov. Também foi desenvolvido um segmentador de aúdio para uso no projeto.

## Dataset

O dataset utilizado foi o AudioMNIST, disponível em https://github.com/soerenab/AudioMNIST. Ele contém os aúdios de 60 falantes repetindo os números de 0 a 9 um total de 50 vezes para cada número, para um total de 30.000 aúdios. A frequência de amostragem dos aúdios é de 48kHz.

Foi feita uma divisão de 80%/20% para o conjunto de treinamento (80%) e teste (20%).

## Pré-processamento

Para o pre-processamento do sinal, foi utilizada a técnica conhecida como *pre-enfâse*, usando um filtro definido por y[n]=x[n]-0.97*x[n-1]. Após isso, foi aplicado um filtro Butterworth passa-baixa de quinta ordem com frequência de corte em 3400Hz.

O principal objetivo dessa etapa é reduzir o ruído e destacar a informação útil do sinal.

## Modelo

O modelo é um classificador que utiliza um conjunto de Modelos Ocultos de Markov (HMMs) treinados para cada palavra do vocabulário. Para a classificação de um aúdio, o modelo calcula as probabilidades de cada HMM ter gerado o aúdio e deduz ser a palavra associada ao HMM com maior probabilidade.

Para as features, utilizou-se os MFCCs (mel-frequency cepstral coefficients) do sinal de aúdio.

## Segmentação

O segmentador de aúdio utiliza intervalos de silêncio para segmentar um aúdio em seções contínuas que estão acima de uma certa faixa de corte. Além disso, a implementação fornecida do segmentador também é responsável por capturar o aúdio do usuário. 

## Especificações

Nessa implementação, foram utilizadas principalmente as bibliotecas *librosa* e *hmmlearn* do *Python*, as *features* utilizadas foram os primeiros 12 MFCCs e cada HMM tinha 5 estados. Para evitar que o modelo ficasse preso em mínimos locais, o treinamento de cada HMM é reinicializado com condições iniciais randômicas um total de 5 vezes.

Para o armazenamento e uso dos arquivos do modelo, foi utilizada a biblioteca *pickle* do Python.

## Resultados

O modelo final (arquivos fornecidos no repositório) teve uma acurácia de cerca de 94% no conjunto de testes. 
