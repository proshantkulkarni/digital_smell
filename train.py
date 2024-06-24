import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat

epochs = 200
lr = 0.001
batch_size = 8

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
model = Model1(11, 32, 1, 17).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad = True)
criterion = nn.CrossEntropyLoss()

train_loss = []
torch.compile(model)
torch.compile(criterion)

for epoch in range(epochs):
  print(f"Epoch {epoch}")

  for X, y in tqdm(loader, leave=False):
    model.train()
    X = X.cuda()
    y = y.cuda()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(dtype = torch.bfloat16):
      output = model(X)
      loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    train_loss.append(loss.item())

  print(f"Loss {loss.item()}")


train_loss = np.reshape(train_loss, (-1, 8))

# epochs = range(200)
# plt.plot(train_loss.reshape(-1), alpha=0.5)
plt.plot(train_loss.mean(axis=1))


acc_list = []

for X, y in loader:
  model.eval()
  X = X.cuda()
  y = y.cuda()
  with torch.no_grad():
    pred = model(X)
  pred = pred.softmax(dim=1)
  pred = torch.max(pred, dim=1).indices

  # print((pred == y).to(torch.float).mean())
  # break
  acc = (pred == y).to(torch.float).mean()
  acc_list.append(acc.item())
  print(acc)


accuracy = np.mean(acc_list)
print(f"Accuracy {accuracy}")