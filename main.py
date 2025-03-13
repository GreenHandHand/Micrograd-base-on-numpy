# type: ignore
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from micrograd.module import SGD, CrossEntropyLoss, Linear, Module
from micrograd.variable import Variable

data = load_digits()
x_train, x_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
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
        self.linear1 = Linear(64, 128)
        self.linear2 = Linear(128, 32)
        self.linear3 = Linear(32, 10)

    def forward(self, x: Variable) -> Variable:
        x = Variable.tanh(self.linear1(x))
        x = Variable.tanh(self.linear2(x))
        # x = Variable.relu(self.linear1(x))
        # x = Variable.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def train_one_epoch(
    model: Module,
    optimizer: SGD,
    loss_fn: Callable,
    x_train: Variable,
    y_train: Variable,
) -> float:
    y_pred = net(x_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.value


def validation(
    model: Module, loss_fn: Callable, x_val: Variable, y_val: Variable
) -> tuple[float, float]:
    y_pred = model(x_val)
    loss = loss_fn(y_pred, y_val)
    acc = np.mean(np.argmax(y_pred.value, axis=1) == y_val.value)
    return loss.value, acc


EPOCH = 1000

net = MLP()
optimizer = SGD(net.parameters(), lr=0.05, weight_decay=4e-3, momentum=0.9)
loss_fn = CrossEntropyLoss()


train_loss_list, val_loss_list, train_step, val_step, acc_list = [], [], [], [], []
with tqdm(total=EPOCH) as pbar:
    for i in range(EPOCH):
        train_loss = train_one_epoch(net, optimizer, loss_fn, x_train, y_train)
        train_loss_list.append(train_loss)
        train_step.append(i)
        pbar.set_postfix(loss=train_loss)
        pbar.update(1)

        if (i + 1) % 10 == 0:
            val_loss, acc = validation(net, loss_fn, x_test, y_test)
            val_loss_list.append(val_loss)
            acc_list.append(acc)
            val_step.append(i)
            print(f"epoch {i} val_loss = {val_loss}, acc = {acc}")


y_pred_test = net(x_test)
loss = loss_fn(y_pred_test, y_test)
print("Test Loss: ", loss.value)
print("Acc: ", np.mean(np.argmax(y_pred_test.value, axis=1) == y_test.value))

ax = plt.subplot(1, 1, 1)
ax_acc = plt.twinx(ax)
ax.plot(train_step, train_loss_list, label="trainLoss")
ax.plot(val_step, val_loss_list, label="valLoss")
ax_acc.plot(val_step, acc_list, label="acc", color="green")
ax.legend()
plt.show()
