import torch
from torch import nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns

# dataset
input_shape = 28
num_classes = 10

# hyper
batch_size = 64
num_epochs = 5
learning_rate = 1e-3

# gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = datasets.MNIST(root='../data/',
                               download=True,
                               train=True,
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='../data/',
                              download=True,
                              train=False,
                              transform=transforms.ToTensor())

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               shuffle=True,
                                               batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              shuffle=False,
                                              batch_size=batch_size)

class CNN(nn.Module):
    def __init__(self, input_shape, in_channels, num_classes):
        super(CNN, self).__init__()
        # conv2d: (b, 1, 28, 28) => (b, 16, 28, 28)
        # maxpool2d: (b, 16, 28, 28) => (b, 16, 14, 14)
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=16,
                                            kernel_size=5, padding=2, stride=1),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))

        # conv2d: (b, 16, 14, 14) => (b, 32, 14, 14)
        # maxpool2d: (b, 32, 14, 14) => (b, 32, 7, 7)
        self.cnn2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32,
                                            kernel_size=5, padding=2, stride=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))
        # (b, 32, 7, 7) => (b, 32*7*7)
        # (b, 32*7*7) => (b, 10)
        self.fc = nn.Linear(32*(input_shape//4)*(input_shape//4), num_classes)


    def forward(self, x):
        # (b, 1, 28, 28) => (b, 16, 14, 14)
        out = self.cnn1(x)
        # (b, 16, 14, 14) => (b, 32, 7, 7)
        out = self.cnn2(out)
        # (b, 32, 7, 7) => (b, 32*7*7)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = CNN(input_shape=input_shape, num_classes=num_classes, in_channels=1).to(device)

device = torch.device('cuda')
model.load_state_dict(torch.load('cnn_mnist.ckpt'))
model.to(device)



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

print(classification_report(y_true, y_pred, digits = 4, target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))



cm = confusion_matrix(y_true, y_pred)
conf_matrix = pd.DataFrame(cm, index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

# plot size setting
fig, ax = plt.subplots(figsize = (20,17))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15}, cmap="Blues")
plt.ylabel('True label', fontsize=15)
plt.xlabel('Predicted label', fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('confusion.pdf', bbox_inches='tight')
plt.show()



