"""
problem tanimi: mnist veri seti ile rakam siniflandirma 
MNIST 
ANN

"""
# library
import torch # tensor işlemleri 
import torch.nn as nn # yapay sinir ağı katmanları tanımlamak için 
import torch.optim as optim # optimizasyon algoritmalarını içeren modul
import torchvision # computer vision and predefined modeleller
import torchvision.transforms as transforms # görüntü dönüşümleri yapmak
import matplotlib.pyplot as plt # görselleştirme 

# optional: cpu ve gpu belirle 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
print("Kullanilan cihaz:" ,device)

# data loading
def get_data_loaders(batch_size=64): # her iterasyonda işlenicek veri miktarı, batch size , memoryi şişirmemek için.

    transform = transforms.Compose([
        transforms.ToTensor(), #görüntüyü tensöre çevirir scaling yapar 0-225->0-1
        transforms.Normalize((0.5,),(0.5,)) # pikselleri -1 ve 1 arası ölçekelr
    ])

    #mnist veri setini indir ve eğitim set kümelerini oluştur 
    train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    #pythorch veri yükleyicisi oluştur
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# data visualization
def visualize_samples(loader, n):
    images, labels = next(iter(loader)) #ilk batchden görüntü ve etiket alma 
    #print(images[0].shape)
    fig, axes = plt.subplots(1, n, figsize=(10,5)) # en farklı görüntü için görselleştirme alanı 
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap= "gray") # görseli gri tamlamalı oalrak göster demek
        axes[i].set_title(f"Label: {labels[i].item()}") # görüntüye ait sınıf etiketini başlık olarak yaz
        axes[i].axis("off") #eksenleri gizle 
    plt.show()    


# define ann model 

# yapay sinir agi class
class NeuralNetwork(nn.Module): # pytorchun nn.module sınıfından miras alıyo 

    def __init__(self): # nninsa etmek için gerekli olan bileşenleri tanımla 
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten() #elimizde bulunan görüntüleri vektör haline ceviricez (1D)

        self.fc1 = nn.Linear(28*28,128) #ilk tam bağlı katmanı olustur 784=input size 128=outputsize

        self.relu= nn.ReLU() #aktivasyon fonk olustur

        self.fc2 = nn.Linear(128,64) #ikinci tam bağlı katmanı olustur 

        self.fc3=nn.Linear(64,10 ) #çıktı katmanı oluştur 64input output size = 10 olamk zorunda (0-9 etiketleri)


    def forward(self,x): # forward propagation: ileri yayılım, giriş olarak x=görüntu alsın
        
        x=self.flatten(x) #initial x=28*28 lijk görüntü flattenla düzleştir 
        x=self.fc1(x) #birinci baglı katman  
        x=self.relu(x) #aktivasyon fonksiyonu 
        x=self.fc2(x) #ikinci baglı katman
        x=self.relu(x) #aktivasyon fonksiyonu 
        x=self.fc3(x) # output katmanı
        return x #modelin çıktısını return edelim 

# create model and compile
model = NeuralNetwork().to(device)

#kayıp fonk ve optimizasyon algoritması belirle
define_loss_and_optimizer= lambda model:(
    nn.CrossEntropyLoss(), #multi class classification problems loss function
    optim.Adam(model.parameters(), lr=0.001) #update weights with adam 
)
 
# train
criterion, optimizer = define_loss_and_optimizer(model)

def train_model(model,train_loader,criterion,optimizer,epochs=10):
    model.train() # modeli egitim moduna al
    train_losses = []  #her bir epoch sonucunda elde edilen loss degerlerini saklamak için bir liste
    for epoch in range (epochs): # belirtilen epoch sayısı kadar eğit
        total_loss = 0

        for images, labels in train_loader:#tüm eğitim verileri üzerinde iterasyon gercekleştir
                images ,labels = images.to(device), labels.to(device) # verileri cihaza tası

                optimizer.zero_grad() #gradyanları sıfırla
                prediction = model(images)#modeli uygula forward pro.
                loss = criterion(prediction, labels) #loss hesap. -> y_prediction ile y_real
                loss.backward()#geri yayılım yani gradyan hesaplama 
                optimizer.step()#update weights

                total_loss =total_loss+ loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.3f}")

    #loss graph 
    plt.figure()
    plt.plot(range(1,epochs+1), train_losses, marker="o", linestyle="-", label= "Train Loss" )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("training Loss")
    plt.legend()
    plt.show()

# test
def test_model(model, test_loader):
    model.eval() #modelimizi degerlendirme moduna al 
    correct =0 # dogru tahmin sayaci
    total =0 #toplam verş sayavi

    with torch.no_grad(): # gardyan hesaplama gereksiz oldugundan kapattık
        for images, labels in test_loader: # test veri kümesini döngüye al
            images, labels = images.to(device), labels.to(device) # verileri cihaza tası
            predictions = model(images) 
            _, predicted = torch.max(predictions, 1)# en yüksek olasılıklı sınıfın etiketini 
        
            total+= labels.size(0) #toplam veri sayısını güncelle 
            correct += (predicted == labels).sum().item() # dogru tahminleri say 
    
    print(f"Test Accuracy: {100*correct / total :.3f} % ")

   

if __name__ == "__main__":
    train_loader, test_loader= get_data_loaders() # veri yukleyicilerini al
    visualize_samples(train_loader,5)

    model= NeuralNetwork().to(device)
    criterion, optimizer= define_loss_and_optimizer(model)

    train_model(model,train_loader, criterion, optimizer)
    test_model(model,test_loader)
