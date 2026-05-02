"""
RNN: tekrarlayan sinir ağları: zaman serilerinde kullanılır 
recurrent neural networks: ardışık verielri işlmek ce zamansal ilişkileri modellemek amacıyla geliştirilmiş sinir ağlarıdır.

kullanım alanları:dogal dil işleme, metin üretimi konuşma tanıma ve finansal zaman serisi analizi gibi zaman bağımlı veri analizi gerektiren alanlarda kullanılırlar.

rnnler bir önceki adımın çıksını kullanırak veriyi işler. buda onlara hafıza kazandırır zamansal ilişkileri öğrenirler. 

"""
#%% veriyi oluştur ve görselleştir 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def generate_data(seq_length = 50, num_samples = 1000):
    """
        example 3lu paket 
        sequence: [2,3,4] giriş dizilerini saklamak için 
        targets: [5]  hedef değerleri saklamak için
    """
    x = np.linspace(0, 100, num_samples)# 0 ile 100 arasında num_samples kadar eşit aralıklı sayı üretir
    y = np.sin(x) 
    sequence = [] # giriş dizilerini saklamak için 
    targets = [] # hedef değerleri saklamak için
    for i in range(len(x) - seq_length):
        sequence.append(y[i:i+seq_length]) # seq_length uzunluğunda ardışık parçalar oluşturur
        targets.append(y[i+seq_length]) # her parçanın sonraki değeri hedef olarak alınır
    return np.array(sequence), np.array(targets) 

sequences, targets = generate_data()

#veriyi görselleştir
plt.figure(figsize=(8,4))
plt.plot(t, y, label='sin(t)', color='blue', linewidth=2)
plt.title('Sinüs Dalga Grafi')
plt.xlabel('Zaman')
plt.ylabel('Genlik')
plt.legend()
plt.grid(True)
plt.show()

#%% rnn modelini oluştur



#%% rnn traning 



#%% rnn test and evaluation
