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

class LSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()# bir üst sınıfın constructor'ını çağırma
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # embedding katmanı
        self.lstm = nn.LSTM(embedding_dim, hidden_dim) # lstm katmanı
        self.fc = nn.Linear(hidden_dim, vocab_size) # tam bağlantılı katman

    def forward(self, x):    # ileri besleme fonksiyonu
        
        x = self.embedding(x) # input-> embedding
        lstm_out, _ = self.lstm(x.view(1,1,-1))
        output=self.fc(lstm_out.view(1,-1)) # lstm-> tam bağlantılı katman
        return output
 
#%%hyperparameter tuning

# kelime listesi -> tensor
def prepare_sequence(seq, to_ix):
    return torch.tensor([to_ix[w] for w in seq], dtype=torch.long)

#hyperparametre tuning kombinasyonlarını belirle
embeding_sizes = [8, 16]# denenecek embedding boyutları
hidden_sizes = [32, 64]# denenecek gizli katman boyutları
learning_rates = [0.01, 0.005]# öğrenme oranı learing rate

best_loss = float('inf') # en düşük kayıp değeri saklamak için bir değişken
best_params={} # en iyi parametreleri saklamak için bir dictionary

print("Hyperparameter Tuning Başlıyor...")

# grid search 
for emb_size, hidden_size, lr in product(embeding_sizes, hidden_sizes, learning_rates):
    print(f"Deneme: Embedding Dim={emb_size}, Hidden Dim={hidden_size}, Learning Rate={lr}")
    
    # model oluşturma
    model = LSTM(len(vocab), emb_size, hidden_size) # seçilen parametrelerlemodel oluşturma
    loss_function = nn.CrossEntropyLoss() # kayıp fonksiyonu
    optimizer = optim.Adam(model.parameters(), lr=lr) # optimizer

    # eğitim 
    epochs= 50
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0 #epoch başına kayıp değerini sıfırla
        for word, next_word in data:
            model.zero_grad() # modelin parametrelerini sıfırla
            input_tensor = prepare_sequence([word], word_to_ix) # girdiyi tensöre çevir
            target_tensor= prepare_sequence ([next_word], word_to_ix) # hedef kelimeyi tensöre çevir
            output = model(input_tensor) # prediction
            loss= loss_function(output, target_tensor)
            loss.backward() # geri yayılım uygula
            optimizer.step() # parametreleri güncelle
            epoch_loss += loss.item() # epoch kaybını güncelle

        if epoch % 10 == 0: # her 10 epochta bir kayıp değerini yazdır
            print(f"Epoch {epoch}, Loss: {epoch_loss:.5f}")    
        total_loss= epoch_loss 

    #en iyi modeli kaydet
    if total_loss < best_loss:
        best_loss = total_loss
        best_params = {'embedding_dim': emb_size, 'hidden_dim': hidden_size, 'learning_rate': lr}
    print()    
print(f"En İyi Parametreler: {best_params}")






#%% lstm training 
final_model= LSTM(len(vocab), best_params['embedding_dim'], best_params['hidden_dim'])
optimizer=optim.Adam(final_model.parameters(), lr = best_params['learning_rate'])
loss_function = nn.CrossEntropyLoss() # kayıp fonksiyonu 
print("final model training")
epochs =100
for epoch in range(epochs):
    epoch_loss=0
    for word, next_word in data:
        final_model.zero_grad()
        input_tensor = prepare_sequence([word], word_to_ix)
        target_tensor= prepare_sequence([next_word], word_to_ix)
        output=final_model(input_tensor)
        loss = loss_function(output, target_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch %10 == 0:
        print(f"final model epoch :{epoch}, loss:{epoch_loss:.5f}")
        


#%% test ve degerlendirme evolution

# kelime tahmini fonsiyonu : başlangıc kelimesini ve n adet kelime uretmesini sağla 
def predict_sequence(start_word, num_words):
    current_word = start_word
    output_sequence = [current_word] # çıktı dizisi

    for _ in range(num_words):# belirtilen sayıda kelime tahmini
        with torch.no_grad(): #gradyan hesaplaması yapmadan
            input_tensor=prepare_sequence([current_word], word_to_ix)# kelimeden tensor dönüşümü 
            output = final_model(input_tensor)
            predicted_idx = torch.argmax(output).item() # en yüksek olasılıga sahip kelimenin indexi
            predicted_word = ix_to_word[predicted_idx] # indexe karsılık gelen kelimeyi return eder
            output_sequence.append(predicted_word)
            current_word = predicted_word # bir sonraki tahmin için mevcut kelimeşeri güncelle 
    return output_sequence # tahmn edilen kelime dizisi return edilir

start_word = "ürün"
num_predictions= 1
predict_sequence= predict_sequence(start_word, num_predictions)
print ("".join(predict_sequence))

# %%
