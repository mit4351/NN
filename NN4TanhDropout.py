##!/usr/nim/python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax, dtanh, dMean_squared_error, mean_squared_error
from PIL import Image
import  matplotlib.pyplot as plt
from common.util import shuffle_dataset
from enum import Enum
import datetime

class Mode(Enum):
      TRAINING = 1
      VALIDATION = 2
      TEST = 3

def forward(x, W1, W2, W3, W4, dropout, training=False):
    a1 = np.dot(x, W1)
    z1 = np.tanh(a1)
    a2 = np.dot(z1, W2)
    z2 = np.tanh(a2)
    # Dropout in layer 2
    if training:
        m2 = np.random.binomial(1, dropout, size=a2.shape)
    else:
        m2 = dropout
    z2 *= m2

    a3 = np.dot(z2, W3)
    z3 = np.tanh(a3)
    # Dropout in layer 3
    if training:
        m3 = np.random.binomial(1, dropout, size=a3.shape)
    else:
        m3 = dropout
    z3 *= m3
    a4 = np.dot(z3, W4)
    y = a4 # linear output
    return z1, z2, z3, y, m2, m3

def backward(x, z1, z2, z3, y, m2, m3, t, W1, W2, W3, W4):
    dC_da4 = dMean_squared_error(y, t)
    dC_dW4 = np.dot(z3.T, dC_da4)
    dC_dz3 = np.dot(dC_da4, W4.T)
    dC_da3 = dC_dz3 * dtanh(z3) * m3
    dC_dW3 = np.dot(z2.T, dC_da3)
    dC_dz2 = np.dot(dC_da3, W3.T)
    dC_da2 = dC_dz2 * dtanh(z2) * m2
    dC_dW2 = np.dot(z1.T, dC_da2)
    dC_dz1 = np.dot(dC_da2, W2.T)
    dC_da1 = dC_dz1 * dtanh(z1)
    dC_dW1 = np.dot(x.T, dC_da1)
    return dC_dW1, dC_dW2, dC_dW3, dC_dW4

def update(W1, W2, W3, W4, dC_dW1, dC_dW2, dC_dW3, dC_dW4, learning_rate):
    # Gradient descent update
    W1 = W1 - learning_rate * dC_dW1
    W2 = W2 - learning_rate * dC_dW2
    W3 = W3 - learning_rate * dC_dW3
    W4 = W4 - learning_rate * dC_dW4
    return W1, W2, W3, W4

def get_data():
    (xx_train, tt_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
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

def init_network(file_name="mit_nn4.pkl"):

    if os.path.isfile(file_name) :
       with open(file_name, 'rb') as f:
           network = pickle.load(f)
    else:
       print(file_name + "存在しない")
       network = init_wait()

    return network

def save_network(network, file_name="mit_nn4.pkl"):
    with open(file_name, 'wb') as f:
         pickle.dump(network, f)

def init_wait():
    n_in = 784
    n_hiddn1 = 100
    n_hiddn2 = 200
    n_hiddn3 = 200
    n_out = 10

    result = {}
    result['W1'] = np.random.uniform(low=-0.1, high=0.1, size=(n_in, n_hiddn1))
    result['W2'] = np.random.uniform(low=-0.1, high=0.1, size=(n_hiddn1, n_hiddn2))
    result['W3'] = np.random.uniform(low=-0.1, high=0.1, size=(n_hiddn2, n_hiddn3))
    result['W4'] = np.random.uniform(low=-0.1, high=0.1, size=(n_hiddn3, n_out))
    return result

def get_sample(X, Y):
    for x, y in zip(X, Y):
        yield x[None,:], y[None,:] # makes sure the inputs are 2d row vectors


def imgi_display(y,t, pimg):
    msg = "誤答"
    print("y=" + str(y))
    print("t=" + str(t))
    p = np.argmax(y)
    print("p=" + str(p))
    a = np.argmax(t)
    print("a=" + str(a))

    if p == a:
       msg = "正答"

    label = "問題=" + str(a) + "   答=" + str(p) + "  (" + msg + ")"
    print(label)
    img = pimg.reshape(28,28)
    plt.imshow(img)
    plt.title(label)
    plt.show()

def testing(network, X_test, Y_test, dropout = 1.0):
    accuracy_cnt = 0
    W1, W2, W3, W4 = network['W1'], network['W2'], network['W3'], network['W4']
    for x, t in get_sample(X_test, Y_test):
        _, _, _, y, _, _ = forward(x, W1, W2, W3, W4, dropout, training=False)
        accuracy_cnt += np.sum(np.argmax(y) == np.argmax(t))
        imgi_display(y, t, x)
    result = float(accuracy_cnt) / len(X_test) * 100
    print("Accuracy:("  + str(accuracy_cnt) + "/"  + str(len(X_test)) + ")="  + str(result) + "%")

def training(network, pX_train, pY_train, X_validation, Y_validation):
    n_epochs = 30
    n_samples =  10000
    #wX, wY = shuffle_dataset(pX_train, pY_train)
    #X_train, Y_train  = wX[:n_samples], wY[:n_samples]
    X_train, Y_train  = pX_train, pY_train
    score_history = []
    score_history2 = []

    W1, W2, W3, W4 = network['W1'], network['W2'], network['W3'], network['W4']
    batch_size = 100
    dropout = 0.5 # 1.0 = no dropout
    learning_rate = 0.01

    best_cost = np.inf
    for epoch in range(n_epochs):
        starttime    =  datetime.datetime.today()
        # Training
        accuracy_cnt = 0
        for x, t in get_sample(X_train, Y_train):
            z1, z2, z3, y, m2, m3 = forward(x, W1, W2, W3, W4, dropout, training=True)
            dC_dW1, dC_dW2, dC_dW3, dC_dW4 = backward(x, z1, z2, z3, y, m2, m3, t, W1, W2, W3, W4)
            W1, W2, W3, W4 = update(W1, W2, W3, W4, dC_dW1, dC_dW2, dC_dW3, dC_dW4, learning_rate)
            accuracy_cnt += np.sum(np.argmax(y) == np.argmax(t))
        result = float(accuracy_cnt) / len(X_train) * 100
        endtime    =  datetime.datetime.today()
        print("Training   Epoch: %3d: Accuracy:(%5d / %5d)= %3.3f %%  %s 〜 %s"  % (epoch+1, accuracy_cnt, len(X_train), result, starttime.strftime("%Y/%m/%d %H:%M:%S"), endtime.strftime("%H:%M:%S")))
        score_history.append(result)

        # Validation
        starttime    =  datetime.datetime.today()
        cost = 0.
        accuracy_cnt2 = 0
        starttime    =  datetime.datetime.today()
        for x, t in get_sample(X_validation, Y_validation):
            _, _, _, y, _, _ = forward(x, W1, W2, W3, W4, dropout, training=False)
            cost += mean_squared_error(y, t)
            accuracy_cnt2 += np.sum(np.argmax(y) == np.argmax(t))
        result2 = float(accuracy_cnt2) / len(X_test) * 100
        #print("Accuracy:("  + str(accuracy_cnt) + "/"  + str(len(X_test)) + ")="  + str(result) + "%")
        endtime    =  datetime.datetime.today()
        print("Validation Epoch: %3d: Accuracy:(%5d / %5d)=(%3.3f)%%  %s 〜 %s"  % (epoch+1, accuracy_cnt2, len(X_test), result2, starttime.strftime("%Y/%m/%d %H:%M:%S"), endtime.strftime("%H:%M:%S")))        #print("Epoch: %d; Cost: %.3f" % (epoch+1, cost))
        score_history2.append(result2)

        if result == 100:
            break

    print("Finished!")
    network['W1'], network['W2'], network['W3'], network['W4'] = W1, W2, W3, W4
    save_network(network, output_filename)
    plotDisplay(score_history, score_history2, len(X_train), len(X_test))

def plotDisplay(score_history, score_history2, tDataCnt, tDataCnt2):
    plt.title("4層NN (Training:%5d  Validation:%5d)"  % (tDataCnt, tDataCnt2))
    plt.plot(score_history, label="Training")
    plt.plot(score_history2, label="Validation")
    plt.legend(loc="lower right")
    plt.xlabel("epoch(" + str(len(score_history)) + ")")
    plt.ylabel("正解率(%)")
    plt.savefig(str(datetime.datetime.today()) + ".png")
    plt.show()

if  __name__ == '__main__':
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    argc = len(argvs) # 引数の個数
    if(argc > 1 and argvs[1] == "ts"):
        mode = Mode.TEST
    else:
        mode = Mode.TRAINING

    output_filename = "mit_nn4tanh.pkl"
    input_filename  = output_filename
    X_test, Y_test, X_train, Y_train,  X_validation, Y_validation  = get_data()
    network = init_network(input_filename)

    if mode == Mode.TRAINING:
        training(network, X_train, Y_train, X_validation, Y_validation)
    else:
        x, t = shuffle_dataset(X_test, Y_test)
        testcount = 10
        testing(network, x[:testcount], t[:testcount])
