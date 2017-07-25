##!/usr/nim/python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from PIL import Image
import  matplotlib.pyplot as plt
from common.util import shuffle_dataset
from enum import Enum


class Mode(Enum):
      TRAINING = 1
      VALIDATION = 2
      TEST = 3

def get_data():
    (xx_train, tt_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

    validation_cnt=10000
    #バリデーションデータ
    x_valid = xx_train[:validation_cnt]
    t_valid = tt_train[:validation_cnt]
    #トレーニングデータ
    x_train = xx_train[validation_cnt:]
    t_train = tt_train[validation_cnt:]

    print('*=*'*25)
    print("*" + ' '*10 + "実行モード (" + str(mode) + ")" )
    print("*" + ' '*10 + "xx_train=" + str(xx_train.shape)  + "   (x_train + x_valid)")
    print("*" + ' '*10 + "x_test  =" + str(x_test.shape)    + "   (テストデータ)")
    print("*" + ' '*10 + "x_train =" + str(x_train.shape)   + "   (トレーニングデータ)")
    print("*" + ' '*10 + "x_valid =" + str(x_valid.shape)   + "   (検証データ)")
    print('*=*'*25)
    return x_test, t_test, x_train, t_train, x_valid, t_valid


def init_network(file_name="mit_ht.pkl"):

    if os.path.isfile(file_name) :
       with open(file_name, 'rb') as f:
           network = pickle.load(f)
    else:
       print(file_name + "存在しない")
       network = init_wait()

    return network

def init_wait(weight_init_std = 0.01):
    #hiddn1 = 50
    #hiddn2 = 100
    #hiddn3 = 100

    hiddn1 = 500
    hiddn2 = 1000

    #hiddn1 = 500
    #hiddn2 = 100
    #hiddn1 = 1000
    #hiddn2 = 1000

    print("hiddn1=" + str(hiddn1))
    print("hiddn2=" + str(hiddn2))

    result = {}
    result['W1'] = weight_init_std * np.random.randn(784, hiddn1) * 0.01
    result['b1'] = np.random.randint(-100,100,hiddn1) * 0.0001
    #result['b1'] = np.zeros(hiddn1)

    result['W2'] = weight_init_std * np.random.randn(hiddn1, hiddn2)  * 0.01
    result['b2'] = np.random.randint(-100,100,hiddn2) * 0.0001
    #result['b2'] = np.zeros(hiddn2)

    result['W3'] = weight_init_std * np.random.randn(hiddn2, 10) * 0.01
    result['b3'] = np.random.randint(-100,100,10) * 0.0001


    return result

def save_network(file_name="mit_ht3.pkl"):
    global W1,W2,W3,network
    global b1,b2,b3
    network['W1'], network['W2'], network['W3'] = W1, W2, W3
    network['b1'], network['b2'], network['b3'] = b1, b2, b3
    with open(file_name, 'wb') as f:
         pickle.dump(network, f)


def imgi_display(a,t, pimg):
    msg = "誤答"
    if a == t:
       msg = "正答"

    label = "問題=" + str(t) + "   答=" + str(a) + "  (" + msg + ")"
    print(label)
    img = pimg.reshape(28,28)
    plt.imshow(img)
    plt.title(label)
    plt.show()

#順伝播
def predict(x):
    global W1,W2,W3
    global b1,b2,b3

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y, z1, z2

# 誤差逆伝播
def back_propagation(x, z1, z2, y, t):
    global W1,W2,W3

    npdata = np.zeros((len(y),10))
    for i in range(len(t)):
        wt = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        wt[t[i]] = 1.0
        npdata[i] = wt

    #3層目のネットワークのパラメータの勾配
    delta3 = y - npdata
    grad_W3 = z2.T.dot(delta3)

    #2層目のネットワークのパラメータの勾配
    sigmoid_dash2 = z2 * (1 - z2)
    delta2 = delta3.dot(W3.T) * sigmoid_dash2
    grad_W2 = z1.T.dot(delta2)

    #1層目のネットワークのパラメータの勾配
    sigmoid_dash = z1 * (1 - z1)
    delta1 = delta2.dot(W2.T) * sigmoid_dash
    grad_W1 = x.T.dot(delta1)

    W3 -= learning_rate * grad_W3
    W2 -= learning_rate * grad_W2
    W1 -= learning_rate * grad_W1


def runing(tr_cnt, x, t, batch_size = 100):
    accuracy_cnt = 0
    disp_cnt = len(x)/10

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]  #[0:100][101:200][......
        t_batch = t[i:i+batch_size]
        y_batch, z1_batch, z2_batch  = predict(x_batch)
        p = np.argmax(y_batch, axis=1) #1次元目の要素ごとに最大値を抽出
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
        back_propagation(x_batch, z1_batch, z2_batch, y_batch, t_batch)

        if mode == Mode.TEST and (i%disp_cnt ==0) :
           imgi_display(p[0], t[i], x[i])
           continue

    result = float(accuracy_cnt) / len(x) * 100
    print("(" + str('%03d' % tr_cnt ) + ")回目   Accuracy:("  + str(accuracy_cnt) + "/"  + str(len(x)) + ")="  + str(result) + "%")
    return result


if  __name__ == '__main__':
    mode = Mode.TRAINING
    #mode = Mode.VALIDATION
    #mode = Mode.TEST
    #input_filename  = "mit_node4_50_100_100.pkl"
    #output_filename = "mit_node4_50_100_100.pkl"
    input_filename  = "mit_node3_500_1000.pkl"
    output_filename = "mit_node3_500_1000.pkl"
    #input_filename  = "mit_node4_500_100_100.pkl"
    #output_filename = "mit_node4_500_100_100.pkl"
    #input_filename  = "mit_node4_1000_1000_1000.pkl"
    #output_filename = "mit_node4_1000_1000_1000.pkl"
    w_cnt_max = 40

    learning_rate = 0.005
    batch_size = 1000
    test_x, test_t, tr_x, tr_t,  v_x, v_t  = get_data()

    #トレーニング
    x = tr_x
    t = tr_t

    if mode == Mode.TEST :
       x, t = shuffle_dataset(test_x, test_t)
       x = x[:1000]
       t = t[:1000]

    if mode == Mode.VALIDATION :
       x, t = v_x, v_t

    print("x=" + str(len(x)))
    print("t=" + str(len(t)))
    print("batch_size=" + str(batch_size))
    print("input  model::" + input_filename)
    print("output model::" + output_filename)

    network = init_network(input_filename)
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    w_cnt=1
    resule=0
    print('-*-'*10 + "トレーニング" + '-*-'*10)
    while resule < 100:
       resule =  runing(w_cnt, x, t)
       w_cnt += 1
       if w_cnt > w_cnt_max:
          break
       if mode != Mode.TRAINING :
          break

    if mode == Mode.TRAINING :
       save_network(output_filename)

#-----------------------
    x, t = v_x, v_t
    print('-*-'*10 + "テスト" + '-*-'*10)
    runing(w_cnt, x, t)
    print('-*-'*20)
