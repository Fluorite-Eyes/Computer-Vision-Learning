from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
from datetime import datetime
from utils import get_mean_and_std
import torch
from torch import nn
from torchvision import models
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
classes = ('plane', 'car' , 'bird',
           'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck')
# dataset
# input_shape = 32
num_classes = 10

# hyper
batch_size = 64
num_epochs = 5
learning_rate = 1e-3

# gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# offline calc mean/std of training dataset
train_dataset = datasets.CIFAR10(root='../data/',
                                 download=True,
                                 train=True,
                                 transform=transforms.ToTensor())
test_dataset = datasets.CIFAR10(root='../data/',
                                download=True,
                                train=False,
                                transform=transforms.ToTensor())
get_mean_and_std(train_dataset)

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])
train_dataset = datasets.CIFAR10(root='../data/',
                                 download=True,
                                 train=True,
                                 transform=transform)
test_dataset = datasets.CIFAR10(root='../data/',
                                download=True,
                                train=False,
                                transform=transform)

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               shuffle=True,
                                               batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              shuffle=False,
                                              batch_size=batch_size)

model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# optimzier = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9,weight_decay=5e-4)
total_batch = len(train_dataloader)

for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # forward
        out = model(images)
        loss = criterion(out, labels)

        # 标准的处理，用 validate data；这个过程是监督训练过程，用于 early stop
        n_corrects = (out.argmax(axis=1) == labels).sum().item()
        acc = n_corrects/labels.size(0)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   # 更细 模型参数

        if (batch_idx+1) % 100 == 0:
            print(f'{datetime.now()}, {epoch+1}/{num_epochs}, {batch_idx+1}/{total_batch}: {loss.item():.4f}, acc: {acc}')

total = 0
correct = 0
confusionMatrix = np.zeros((10, 10))


y_pred = []
y_true = []
for images, classed in test_dataloader:

    images = images.to(device)
    classed = classed.to(device)
    out = model(images)
    preds = torch.argmax(out, dim=1)
    y_pred = np.hstack((y_pred, preds.cpu()))
    y_true = np.hstack((y_true, classed.cpu()))
    total += images.size(0)
    correct += (preds == classed).sum().item()

print(classification_report(y_true, y_pred, digits = 4, target_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']))



cm = confusion_matrix(y_true, y_pred)
conf_matrix = pd.DataFrame(cm, index=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], columns=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

# plot size setting
fig, ax = plt.subplots(figsize = (20,17))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15}, cmap="Blues")
plt.ylabel('True label', fontsize=15)
plt.xlabel('Predicted label', fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('confusion.pdf', bbox_inches='tight')
plt.show()



