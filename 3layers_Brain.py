##!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def softmax(u):
    e = np.exp(u)
    return e / np.sum(e)

# 順伝播
def forward(x):
    global W1
    global W2
    global W3
    u1 = x.dot(W1)
    z1 = sigmoid(u1)
    u2 = z1.dot(W2)
    z2 = sigmoid(u2)
    u3 = z2.dot(W3)
    y = softmax(u3)
    return y, z1, z2

# 誤差逆伝播
def back_propagation(x, z1, z2, y, d):
    global W1
    global W2
    global W3
    #softmaxの交差エントロピー
    delta3 = y - d
    print('*'*20 + '(back_propagation)' + '*'*20)
    print('*'*20 + '(y)' + '*'*20)
    print(y)
    print('*'*20 + '(d)' + '*'*20)
    print(d)


    print('*=*'*40)
    print('*'*10 + '(z2)' + '*'*10)
    print(z2.T)
    print('*'*10 + '(delta3)' + '*'*10)
    print(delta3)
    print('*=*'*40)


    #3層目のネットワークのパラメータの勾配
    grad_W3 = z2.T.dot(delta3)
    #2層目のネットワークのパラメータの勾配
    sigmoid_dash2 = z2 * (1 - z2)
    delta2 = delta3.dot(W3.T) * sigmoid_dash2
    grad_W2 = z1.T.dot(delta2)

    sigmoid_dash = z1 * (1 - z1)
    delta1 = delta2.dot(W2.T) * sigmoid_dash
    #1層目のネットワークのパラメータの勾配
    grad_W1 = x.T.dot(delta1)

    print(learning_rate)
    W3 -= learning_rate * grad_W3
    W2 -= learning_rate * grad_W2
    W1 -= learning_rate * grad_W1



W1 = np.array([[0.1, 0.3], [0.2, 0.4]])
W2 = np.array([[0.1, 0.3], [0.2, 0.4]])
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])

print(W1)
learning_rate = 0.005

# 順伝播
x = np.array([[1, 0.5]])
y, z1, z2 = forward(x)

print('*'*20 + '(z2)' + '*'*20)
print(z2)
print('*'*20 + '(z1)' + '*'*20)
print(z1)
print('*'*20 + '(y)' + '*'*20)
print(y)

# 誤差逆伝播
# 教師データ
d = np.array([[1, 0]])
back_propagation(x, z1, z2, y, d)


print('*'*20 + '(W3)' + '*'*20)
print(W3)
print('*'*20 + '(W2)' + '*'*20)
print(W2)
print('*'*20 + '(W1)' + '*'*20)
print(W1)
print('*'*40)
