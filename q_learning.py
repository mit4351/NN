##!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
'''
特定の文字列(単語)に次のように得点を割り当てて、
{“A”: 1, “BB”: 1, “AB”: 2, “ABB”: 3, “BBA”: 3, “BBBB”: 4}
文字列中に含まれる単語の合計得点を文字列の得点とするということにします。
例えば”AAAAAAAA”なら8点(1 * 8)、”AAAAAAAB”なら9点(1 * 7 + 2)です。

環境のとりうる状態は長さ8のそれぞれに’A’, ‘B’の2通りあるので2^8通りあり、
アクションは、何もしないのと、位置1~8のそれぞれを’A’または’B’なのでを計17通りあります。
今回のコードでは状態は文字列、アクションは整数(0~16)で管理します。


環境の状態は、'A'または'B'からなる長さ8の文字列で表され、
その文字列にはある法則により得点がつけられる。
プレイヤーはその法則についての知識を予め持たないが、
文字列中の任意の1文字を選んで'A'または'B'に置き換えることができ、
その結果、その操作による得点の変化量を報酬として受け取る。

'''

STR_LEN = 8
ALPHA = 0.1
GAMMA = 0.99
EPS = 0.1


def int_to_state(n):
    ret = ""
    for i in range(STR_LEN):
        if (n >> i) & 1:
            ret += "B"
        else:
            ret += "A"
    return ret


def score(s):
    point_dict = {"A": 1, "BB": 1, "AB": 2, "ABB": 3, "BBA": 3, "BBBB": 4}
    ret = 0
    for i in range(STR_LEN):
        for j in range(i, STR_LEN):
            if s[i:j+1] in point_dict:
                ret += point_dict[s[i:j+1]]
    return ret


def do_action(s, a):
    current_score = score(s)
    if a == 0:
        next_s = s
    elif 1 <= a <= STR_LEN:
        next_s = s[:a-1] + "A" + s[a:]
    else:
        next_s = s[:a-STR_LEN-1] + "B" + s[a-STR_LEN:]
    reward = score(next_s) - current_score
    return next_s, reward


def QLearning(n_rounds, t_max, cnt=1):
    # init
    s0 = "A" * STR_LEN  #'AAAAAAAA'
    A = range(STR_LEN*2+1) #位置1~8のそれぞれを’A’または’B’なのでを計17通
    Q = {}
    #環境のとりうる状態は長さ8のそれぞれに’A’, ‘B’の2通りあるので2^8通
    for i in range(2**STR_LEN):
        Q[int_to_state(i)] = np.random.random(len(A))
    #2^8(256)通全てのパタンーに17通アクションの初期重みを設定
    score_history = []

    for i in range(n_rounds):
        s = s0
        # learn
        for t in range(t_max):
            # select action
            if np.random.random() < EPS:
                a = Q[s].argmax()
            else:
                a = np.random.choice(A)
            # do action next_s:次の状態 reward:スコア増加値
            next_s, reward = do_action(s, a)
            # update Q, and change s
            Q[s][a] += ALPHA * (reward + GAMMA*Q[next_s].max() - Q[s][a])
            s = next_s

        # test
        s = s0
        for t in range(t_max):
            a = Q[s].argmax()
            s, _ = do_action(s, a)
        score_history.append(score(s))

    # visualize
    plt.plot(score_history)
    plt.show()

if __name__ == '__main__':
    #np.random.seed(123)
    QLearning(1000, 20)
    #plt.show()
