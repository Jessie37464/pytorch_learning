import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


x = torch.linspace(-5, 5, 100)
x = Variable(x)

x_np = x.data.numpy()

y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()


plt.figure(1, figsize=(8, 6))

plt.subplot(221)
plt.plot(x_np, y_relu, c='r', label='relu')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='g', label='sigmoid')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='b', label='tanh')

plt.show()