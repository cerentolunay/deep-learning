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
    
    #veriyi görselleştir
    plt.figure(figsize=(8,4))
    plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
    plt.title('Sinüs Dalga Grafi')
    plt.xlabel('Zaman')
    plt.ylabel('Genlik')
    plt.legend()
    plt.grid(True)
    plt.show()
    return x,y,np.array(sequence), np.array(targets) 

x, y,sequences, targets = generate_data()

#%% rnn modelini oluştur
class RNN(nn.Module):
    def __init__(self, input_size,output_size, hidden_size, num_layers=1):
        """
        rnn tanımla 
        linear (output)
        """

        super(RNN,self).__init__() 
        #input_size : gitiş boyutu, hidden_size: gizli katman cell boyutu, num_layers: rnn katmanı sayısı, batch_first: giriş ve çıkış tensörlerinin ilk boyutunun batch boyutu olduğunu belirtir 
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)#rnn katmanı 
        #output_size cıktı boyutu 1
        self.fc=nn.Linear(hidden_size, output_size) # fully connected layer: output


    def forward(self, x):

        out, _ = self.rnn(x) # rnn e girdiyi ver çıktıyı al 
        out = self.fc(out[:, -1, :]) # rnn çıktısının son zaman adımını al ve fully connected layer a ver
        
        return out

model = RNN(1,1,16,1)


#%% rnn traning 
"""
hyperparameters: öğrenme oranı, epoch sayısı, batch size gibi modelin eğitim sürecini kontrol eden parametrelerdir.
"""

# hyperparameters
seq_length = 50 # input dizisinin boyutu
input_size = 1 # input dizisinin her bir elemanının boyutu
hidden_size = 16 # rnnin gizli katmanındaki düğüm sayısı
output_size = 1 # output boyutu ya da tahmin edilen değer 
num_layers = 1 # rnn katmanı sayısı
epochs = 20 # modelin eğitim süreci boyunca tüm eğitim verisinin kaç kez kullanılacağı
batch_size = 32 # her bir eğitim adımında kaç örneğin kullanılacağı
learning_rate = 0.001 # optimizasyon algoritması için öğrenme oranı ya da hızı 

# veriyi hazırla 
x, y, sequences, targets = generate_data(seq_length)

# RNN'e verilecek gerçek giriş ve hedef değerler
x_train = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1) # sequences -> tensor ve son boyuta input_size ekle
y_train = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1) # targets -> tensor ve çıktı boyutu ekle

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

dataset = torch.utils.data.TensorDataset(x_train, y_train) # veri seti oluştur
dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) # veri yükleyici oluştur

# modeli tanımla 
model = RNN(input_size, output_size, hidden_size, num_layers) # rnn modelini oluştur
criterion = nn.MSELoss() # kayıp fonksiyonu: mean square error - ortalama kare hata
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # optimizasyon = adaptive momentum

for epoch in range(epochs):
    for batch_x, batch_y in dataLoader:
        optimizer.zero_grad() # gradyanları sıfırla

        outputs = model(batch_x) # modeli çalıştır ve tahminleri al
        loss = criterion(outputs, batch_y) # kayıp hesapla

        loss.backward() # gradyanları geri yay
        optimizer.step() # model parametrelerini güncelle
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}') # her epoch sonunda kayıp değerini yazdır#%% rnn test and evaluation


# %% rnn test and evaluation

# %% rnn test and evaluation

# %% rnn test and evaluation

# iki adet test verisi oluştur
x_test = np.linspace(100, 110, seq_length).reshape(-1, 1)   # ilk test verisi
x_test2 = np.linspace(120, 130, seq_length).reshape(-1, 1)  # ikinci test verisi

y_test = np.sin(x_test)     # ilk test verisinin sinüs değerleri
y_test2 = np.sin(x_test2)   # ikinci test verisinin sinüs değerleri

# from numpy to tensor
x_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(1, seq_length, input_size)
x_test2_tensor = torch.tensor(y_test2, dtype=torch.float32).reshape(1, seq_length, input_size)

# modeli kullanarak prediction yap
model.eval()

with torch.no_grad():
    prediction1 = model(x_test_tensor).detach().numpy()
    prediction2 = model(x_test2_tensor).detach().numpy()

# sonuçları görselleştir
plt.figure(figsize=(12, 5))

# training dataset
plt.plot(
    np.linspace(0, 100, len(y)),
    y,
    marker='o',
    markersize=3,
    label='Training dataset'
)

# test dataları index mantığıyla çiziliyor
test_axis = np.arange(seq_length)

plt.plot(
    test_axis,
    y_test.flatten(),
    marker='o',
    label='Test 1'
)

plt.plot(
    test_axis,
    y_test2.flatten(),
    marker='o',
    label='Test 2'
)

# prediction noktaları 50. adıma koyuluyor
plt.plot(
    seq_length,
    prediction1.flatten()[0],
    "ro",
    label='Prediction 1',
    markersize=8
)

plt.plot(
    seq_length,
    prediction2.flatten()[0],
    "ro",
    label='Prediction 2',
    markersize=8
)

plt.title("RNN Sinüs Tahmini")
plt.xlabel("Zaman")
plt.ylabel("Genlik")
plt.legend()
plt.grid(True)
plt.show()

print("prediction1:", prediction1.flatten()[0])
print("prediction2:", prediction2.flatten()[0])
# %%
