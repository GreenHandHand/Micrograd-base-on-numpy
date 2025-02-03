import numpy as np
from micrograd.variable import Variable
from micrograd.module import Linear, CrossEntropyLoss, SGD, Module

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from dataclasses import dataclass
from tqdm import tqdm

data = load_digits()
x_train, x_test, y_train, y_test = train_test_split(
    data.data,  # type: ignore
    data.target,  # type: ignore
    test_size=0.2,
)

x_train = Variable(x_train)
x_test = Variable(x_test)
y_train = Variable(y_train)
y_test = Variable(y_test)

print(x_train.shape, y_train.shape)


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(64, 32)
        self.linear2 = Linear(32, 16)
        self.linear3 = Linear(16, 10)

    def forward(self, X: Variable) -> Variable:
        X = Variable.tanh(self.linear1(X))
        X = Variable.tanh(self.linear2(X))
        X = self.linear3(X)
        return X


EPOCH = 10000

net = MLP()
optimizer = SGD(net.parameters(), lr=0.01, weight_decay=4e-3, momentum=0.9)
loss_fn = CrossEntropyLoss()
with tqdm(total=EPOCH) as pbar:
    for epoch in range(EPOCH):
        y_pred = net(x_train)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.value)
        pbar.update()

y_pred_test = net(x_test)
loss = loss_fn(y_pred_test, y_test)
print("Test Loss: ", loss.value)
print("Acc: ", np.mean(np.argmax(y_pred_test.value, axis=1) == y_test.value))
