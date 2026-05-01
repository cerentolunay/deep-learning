"""
problem tanimi: sınıflandırma problemi CIFAR10 veri seti kullanarak bir CNN modeli oluşturacağız. CNN modeli oluştururken aşağıdaki adımları takip edeceğiz:

"""
# %% import libraries 
import torch # pythorch library
import torch.nn as nn # neural netrok library
import torch.optim as optim  # optimization library 
import torchvision # computer vision library
import torchvision.transforms as transforms # data augmentation library
import matplotlib.pyplot as plt # visualization library  
import numpy as np # numerical computing library  
import os      

# load dataset  
def get_dataloaders(batch_size=64): #batch size her iterasyonda islenecek veri sayısını belirler

    transform = transforms.Compose([
        transforms.ToTensor(), # convert images to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # rgb kanallarını normalize eder 
    ])
# CIFAR10 veri setini indir ve eğitim ve test kümelerini oluştur
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True,download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# PyTorch veri yükleyicisi oluştur
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

#%% visualeize dataset

def imshow(img):
    # verileri normalize etmeden önce geri dönüştür
    img = img / 2 + 0.5 # normalize edilmiş görüntüyü geri ölçeklendir
    np_img = img.numpy() # tensörden numpy arrayine çevir
    plt.imshow(np.transpose(np_img, (1, 2, 0))) # 3 kanal için renkelri dogru sekilde gösterme 
   

def get_sample_images(train_loader): #veri kümesinden ornek görselleri alır
    dataiter = iter(train_loader) # veri yükleyicisinden bir iterator oluştur
    images, labels = next(dataiter) # ilk batchten görüntüleri ve etiketleri al
    return images, labels

def visualize(n=5):
    train_loader, test_loader = get_dataloaders() # veri yükleyicilerini al
    # n tane veri gorsellestirme 
    images, labels = get_sample_images(train_loader) # örnek görselleri al
    plt.figure()
    for i in range(n):
        plt.subplot(1, n, i+1) # görselleştirme alanını oluştur
        imshow(images[i]) # görselleri göster
        plt.title(f"Label: {labels[i].item()}") # görsellere ait etiketleri başlık olarak yaz
        plt.axis("off") # eksenleri gizle
    plt.show()

#visualize(3)

#%% build CNN model
class CNN(nn.Module): # CNN sınıfı nn.Module sınıfından miras alır

    def __init__(self):
        
        super(CNN, self).__init__() # nn.Module sınıfının init metodunu çağır
        # convolutional katmanları tanımla
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) # ilk konvolüsyon katmanı 3 giriş kanalı (RGB), 32 çıkış kanalı, 3x3 kernel boyutu
        self.relu =nn.ReLU() # aktivasyon fonksiyonu (hızlı çalışır)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # max pooling katmanı 2x2 kernel boyutu ve stride 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # ikinci konvolüsyon katmanı 32 giriş kanalı, 64 çıkış kanalı, 3x3 kernel boyutu
        self.dropout = nn.Dropout(0.2) # dropout katmanı %20 oranında nöronları rastgele kapatır

        self.fc1 = nn.Linear(64*8*8, 128) # fully connected layer giriş=4096 (64 kanal * 8x8 boyut), çıkış=128
        self.fc2 = nn.Linear(128, 10) # fully connected layer giriş=128, çıkış=10 (sınıf sayısı)

        #image 3*32*32 -> conv(32) -> relu(32) -> pool (16)
        #conv(16) -> relu(16) -> pool (8) ->image = 8*8


    def forward(self, x):
        """
        image 3*32*32 -> conv(32) -> relu(32) -> pool (16)
        conv(16) -> relu(16) -> pool (8) ->image = 8*8
        flatten 
        fc1 -> relu -> dropout -> fc2 -> output
        """
        x = self.pool(self.relu(self.conv1(x))) # ilk convolution block
        x = self.pool(self.relu(self.conv2(x))) # ikinci convolution block
        x = x.view(-1, 64*8*8) # konvolüsyon katmanlarının çıktısını flatten yap
        x = self.dropout(self.relu(self.fc1(x))) # fully connected layer 
        x = self.fc2(x) # output layer
        return x  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cihazı belirle (GPU varsa kullan, yoksa CPU kullan)    
#model = CNN().to(device) 

# define loss function and optimizer
define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), # multi-class classification problems için loss function
    optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # stochastic gradient descent optimizer with momentum 
)


#%% train the model

def train_model(model, train_loader, criterion, optimizer, epochs = 5):
    model.train()# modeli eğitim moduna al
    train_losses = [] #loss degerlerini saklamak için bir liste olsutur
    
    for epoch in range(epochs): #belirtilen epoch sayısı için for döngüsü oluştur
        total_loss = 0 #toplam loss değerini saklamak içim total_loss 
        for images, labels in train_loader: #for loop tüm eğitim veri setini taramak için 
            images, labels = images.to(device), labels.to(device) #verileri cihaza gönder (GPU veya CPU)
            optimizer.zero_grad() #gradyanları sıfırla
            outputs = model(images) # forward pro.(prediction) yap
            loss = criterion(outputs, labels) # loss degeri hesapla
            loss.backward() # backward pro.(gradyan hesapla) yap
            optimizer.step() # öğrenme = ağırlıkları güncelle

            total_loss += loss.item() # toplam loss değerini güncelle
    
        avg_loss = total_loss / len(train_loader) # ortalama loss değerini hesapla
        train_losses.append(avg_loss) # ortalama loss değerini listeye ekle
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.5f}") # her epoch sonunda loss değerini yazdır

    # loss grafiği çiz
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker="o", linestyle="-", label="Train Loss") # loss değerlerini grafiğe ekle
    plt.xlabel("Epochs") # x ekseni etiketi
    plt.ylabel("Loss") # y ekseni etiketi
    plt.title("Training Loss") # grafik başlığı
    plt.legend() # legend göster
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_loss.png",dpi=300, bbox_inches="tight")
    plt.show() # grafiği göster
    plt.close() # grafiği kapat
#train_loader, test_loader = get_dataloaders()
#model= CNN().to(device)
#criterion, optimizer = define_loss_and_optimizer(model)
#train_model(model, train_loader, criterion, optimizer, epochs=10)


#%% test the model
# model ne kadar öğrendi ve test veri setinde ne kadar başarılı olduğunu görmek için test aşamasına geçelim

def test_model(model, test_loader, dataset_type="test"):

    model.eval() # modeli değerlendirme moduna al
    correct = 0 # doğru tahmin sayacı
    total = 0 # toplam tahmin sayacı

    with torch.no_grad(): # gradyan hesaplamalarını kapat
        for images, labels in test_loader: #test veri setiyle değerlendirme 
            images, labels = images.to(device), labels.to(device) 
            outputs = model(images) # prediction 
            _, predicted = torch.max(outputs, 1) # en yüksek olasılığa sahip sınıfı tahmin olarak al
            total += labels.size(0) #toplam tahmin sayısını güncelle
            correct += (predicted == labels).sum().item() #doğru tahmin sayısını güncelle

    accuracy = 100 * correct / total
    print(f"{dataset_type} accuracy: {accuracy:.2f}%")

    os.makedirs("results", exist_ok=True)

    with open("results/results.txt", "a", encoding="utf-8") as f:
        f.write(f"{dataset_type} accuracy: {accuracy:.2f}%\n")
    return accuracy
#test_model(model, test_loader, dataset_type="test") #test accuracy 62.58 %
#test_model(model, train_loader, dataset_type="training") # training accuracy 65.378 % 





# %% main program 

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    open("results/results.txt", "w").close()

    #veri seti yükleme
    train_loader, test_loader = get_dataloaders() 
    #görselleştirme
    visualize(5)

    #training
    model = CNN().to(device) 
    criterion, optimizer = define_loss_and_optimizer(model) 
    train_model(model, train_loader, criterion, optimizer, epochs=10) 

    # test
    test_accuracy = test_model(model, test_loader, dataset_type="test")
    train_accuracy = test_model(model, train_loader, dataset_type="training")




# %%
