import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor())

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor())

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device", device)


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 3, 3),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3)
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1452, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        conv_output = self.conv_stack(x)
        linear_input = self.flatten(conv_output)
        # logits is essentially inverse of probability squashing function (often sigmoid)
        logits = self.linear_relu_stack(linear_input)
        return logits


model = NeuralNet().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

n_epochs = 10

for i in range(n_epochs):
    epoch_loss = 0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print("epoch", i, "loss:", epoch_loss)

test_batch, label = next(iter(test_dataloader))

img = test_batch[0]

with torch.no_grad():
    print(model(torch.unsqueeze(img, 0)))
    print("true label:", label[0])
    plt.imshow(img.numpy().squeeze(), cmap="gray_r")
    plt.show()
