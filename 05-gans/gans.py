"""
image generation: MNIST veri seti kullanacağız

Generative Adversarial Networks (çekişmeli üretici ağlar) birbirine karşi çalışan iki yapay sinir ağı kullanılarak eğitilen bir derin öğrenme modelidir.
Bu ağlardan biri veri üretir(üretici), diğeri ise üretilen verilerin gerçekmi yoksa sahte mi olduğunu değerlendir(ayrıt edici)
örnek: bir gan modeli, yüksek kaliteli ve gerçekçi insan yüzü fotoğrafı üretmek için kullanılabilir
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as utils 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import numpy as np 

# %% veri seti hazırlama 
device  = torch.device("cuda" if torch.cuda.is_available() else "c")

batch_size = 128 # mini batch boyutu
image_size = 28*28 # görüntü boyutu

trasnform = transforms.Compose([
    transforms.ToTensor(), # görüntüleri tensore cevir
    transforms.Normalize((0.5,),(0.5,))# normalizasyon -> -1 ile 1 arasında sıkıştır
])

#MNIST veri seti yükleme
dataset = datasets.MNIST(root= "./data_mnist", train= True, transform=transform, dowland = True)

# veri setini batcler halinde yükle
dataLoader= DataLoader(dataset, batch_size= batch_size, shuffle= True)

#%% discriminator oluştur

class Discriminator(nn.Module): #Ayırt edici: generatorun urettiği görüntülerin sahte mi gercekmi oldugunbu anlamaya çalışacak

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model= nn.Sequential(
            nn.Linear(image_size, 1024), # input: image size, 1024: nöron sayısı yani nu layerin outputu
            nn.LeakyReLU(0.2), # aktivasyon fonk ve 0.2lik eğim 
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2), 
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1), # tek bir cıktı gercek mi sahte mi
            nn.Sigmoid() # çıktıyı 0-1 arasına getir
        )
            
    def forward(self,img):
        return self.model(img.view(-1, image_size)) # görüntüyü düzleştirerek modele ver


#%% generator oluştur 

#%% gan training 

# %% model testing and performance evaluation