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
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)



model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(train_dataloader)
for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # forward
        out = model(images)
        loss = criterion(out, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   # 更细 模型参数

        if (batch_idx+1) % 100 == 0:
            print(f'{epoch+1}/{num_epochs}, {batch_idx+1}/{total_batch}: {loss.item():.4f}')



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



