##!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def softmax(u):
    e = np.exp(u)
#    print('*'*20 + 'softmax(u)' + '*'*20)
#    print('*'*15 + 'u' + '*'*15)
#    print(u)
#    print('*'*15 + 'e' + '*'*15)
#    print(e)
#    print('*'*15 + 'np.sum(e)' + '*'*15)
#    print(np.sum(e))
#    print('*'*15 + 'e / np.sum(e)' + '*'*15)
#    print(e / np.sum(e))
    return e / np.sum(e)

# 順伝播
def forward(x):
    global W1
    global W2
    u1 = x.dot(W1)
    z1 = sigmoid(u1)
    u2 = z1.dot(W2)
    y = softmax(u2)
    return y, z1

# 誤差逆伝播
def back_propagation(x, z1, y, d):
    global W1
    global W2
    #softmaxの交差エントロピー
    delta2 = y - d
    #2層目のネットワークのパラメータの勾配
    grad_W2 = z1.T.dot(delta2)

    sigmoid_dash = z1 * (1 - z1)
    delta1 = delta2.dot(W2.T) * sigmoid_dash
    #1層目のネットワークのパラメータの勾配
    grad_W1 = x.T.dot(delta1)

    W2 -= learning_rate * grad_W2
    W1 -= learning_rate * grad_W1



W1 = np.array([[0.1, 0.3], [0.2, 0.4]])
W2 = np.array([[0.1, 0.3], [0.2, 0.4]])
learning_rate = 0.005

# 順伝播
x = np.array([[1, 0.5]])
y, z1 = forward(x)

print('*'*20 + '(z1)' + '*'*20)
print(z1)
print('*'*20 + '(y)' + '*'*20)
print(y)

# 誤差逆伝播
# 教師データ
d = np.array([[1, 0]])
back_propagation(x, z1, y, d)


print('*'*20 + '(W1)' + '*'*20)
print(W1)
print('*'*20 + '(W2)' + '*'*20)
print(W2)
print('*'*40)
