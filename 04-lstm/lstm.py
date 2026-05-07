#lstm nedir? rnnlerin bellek sorununu çözmek için geliştirilmiş bir tür yinelemeli sinir ağıdır.
# dil modelleri, metin analizi, otomatik metin tamamlama ve hisse senedi fiyat tahmini gibi birçok uygulamada kullanılırlar. 
# lstmde oneli gateler: forget gate input gate output gate

'''
problem tanımı: lstm ile metin türetme 

'''
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter # kelime frekanslarını hesaplamak için 
from itertools import product # grid search için kombinasyonları oluşturmak 

# %% veri yükleme ve preprocessing
# urun yorumları
text = """Bu ürün beklentimi fazlasıyla karşıladı.
Malzeme kalitesi gerçekten çok iyi. 
Kargo hızlı ve sorunsuz bir şekilde elime ulaştı. 
Fiyatına göre performansı harika. 
Kesinlikle tavsiye ederim ve öeneririm!"""

# veri ön işleme: 
# noktalama işaretlerini kaldırma
# kelimeleri küçük harfe çevirme
# kelimeleri böl

words = text.replace(".", "").replace("!", "").lower().split() 

# kelime frekanslarını hesaplama ve indeksleme
word_counts = Counter(words)
vocab = sorted(word_counts, key=word_counts.get, reverse=True) 
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}      


# eğitim verisi hazırlama 
data = [(words[i], words[i+1]) for i in range(len(words)-1)]
 




#%% lstm modeli tanımlama




#%%hyperparameter tuning


#%% lstm training 


#%% test ve degerlendirme evolution



