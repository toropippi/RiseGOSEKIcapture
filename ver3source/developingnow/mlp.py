import time
import numpy as np
import random
from PIL import Image
import os
np.set_printoptions(suppress=True)#強制的に小数表記


# Batch Normalizationの実装
class BatchNormalization:

    # インスタンス変数の定義
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        # 再変換用のパラメータ
        self.gamma = gamma  # 標準偏差
        self.beta = beta  # 平均
        self.momentum = momentum  # 減衰率

        # テスト時に使用する統計量
        self.running_mean = running_mean  # 平均
        self.running_var = running_var  # 分散

        # 逆伝播時に使用する統計量
        self.batch_size = None  # データ数
        self.xc = None  # 偏差
        self.std = None  # 標準偏差
        self.dgamma = None  # (再変換用の)標準偏差の微分
        self.dbeta = None  # (再変換用の)平均の微分

    # 順伝播メソッドの定義
    def forward(self, x, train_flg=True):
        # 初期値を与える
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:  # 学習時
            # 正規化の計算
            mu = x.mean(axis=0)  # 平均
            xc = x - mu  # 偏差
            var = np.mean(xc ** 2, axis=0)  # 分散
            std = np.sqrt(var + 10e-7)  # 標準偏差
            xn = xc / std  # 標準化:式(6.7)

            # 計算結果を(逆伝播用に)インスタンス変数として保存
            self.batch_size = x.shape[0]
            self.xc = xc  # 偏差
            self.xn = xn  # 標準化データ
            self.std = std  # 標準偏差
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu  # 過去の平均の情報
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var  # 過去の分散の情報
        else:  # テスト時
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + 10e-7)  # 標準化:式(6.7')

        # 再変換
        out = self.gamma * xn + self.beta  # 式(6.8)
        return out

    # 逆伝播メソッドの定義
    def backward(self, dout):
        # 微分の計算
        dbeta = dout.sum(axis=0)  # 調整後の平均
        dgamma = np.sum(self.xn * dout, axis=0)  # 調整後の標準偏差
        dxn = self.gamma * dout  # 正規化後のデータ
        dxc = dxn / self.std  # 偏差
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)  # 標準偏差
        dvar = 0.5 * dstd / self.std  # 分散
        dxc += (2.0 / self.batch_size) * self.xc * dvar  # 偏差
        dmu = np.sum(dxc, axis=0)  # 平均
        dx = dxc - dmu / self.batch_size  # 入力データ

        # インスタンス変数に保存
        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx

    # alphaで係数修正
    def modyfyGB(self, alpha):
        self.gamma -= alpha * self.dgamma
        self.beta -= alpha * self.dbeta
        return



# ReLU関数 ------------------------
def ReLU(x):
    return np.maximum(x, 0.0)

def retX2(x_train):
    N, D = x_train.shape
    x_train2 = np.zeros((N, D + 1),dtype=np.float32)  # N*(D+1)
    x_train2[:, 0:D] = x_train[:, 0:D]
    x_train2[:, D] = 1.0
    return x_train2

# ネットワーク  ------------------------
def FNN(wv, M, M2, K, x,bn1 ,bn2,train_flg):
    N, D = x.shape
    w = wv[0:M * D] # 中間層ニューロンへの重み
    w = w.reshape(M, D)
    w2 = wv[M * D:M * D + M2 * M]  # 中間層ニューロンへの重み
    w2= w2.reshape(M2, M)
    v = wv[M * D + M2 * M:]  # 出力層ニューロンへの重み
    v = v.reshape(K, M2)
    b = np.zeros((N, M),dtype=np.float32)  # 中間層ニューロンの入力総和
    z = np.zeros((N, M),dtype=np.float32)  # 中間層ニューロンの出力
    b2 = np.zeros((N, M2),dtype=np.float32)  # 中間層ニューロンの入力総和
    z2 = np.zeros((N, M2),dtype=np.float32)  # 中間層ニューロンの出力
    a = np.zeros((N, K),dtype=np.float32)  # 出力層ニューロンの入力総和
    y = np.zeros((N, K),dtype=np.float32)  # 出力層ニューロンの出力

    b = np.dot(x, w.T)  # N*M
    outb = bn1.forward(b, train_flg)
    z = ReLU(outb)

    b2 = np.dot(z, w2.T)  # N*M2
    outb2 = bn2.forward(b2, train_flg)
    z2 = ReLU(outb2)

    a = np.dot(z2, v.T)  # N*K
    amx = np.max(a, axis=1)
    amx = (amx[:] > 30.0) * (amx[:]-30.0)
    amx = amx.reshape((amx.shape[0], 1))
    a = a - amx

    expa = np.exp(a)
    wkz = np.sum(expa,axis=1)
    rwkz = 1.0 / wkz
    for n in range(rwkz.shape[0]):
        y[n,:] = expa[n,:] * rwkz[n]
    return y, a, z, z2, b, b2,outb,outb2

def CE_FNN(wv, M,M2, K, x, t,bn1 ,bn2,train_flg):
    N, D = x.shape
    y, a, z,z2, b,b2,outb,outb2 = FNN(wv, M,M2, K, x,bn1 ,bn2,train_flg)

    yrsp = y.reshape(-1)
    yrsp[(yrsp[:] <= 0.0)] = 0.000000000000001
    nplgy = np.log(yrsp)
    ce = -np.dot(nplgy, t.reshape(-1)) / N
    return ce

def dCE_FNN(wv, M,M2, K, x, t, bn1, bn2):
    N, D = x.shape
    # wv を w と v に戻す
    w = wv[0:M * D] # 中間層ニューロンへの重み
    w = w.reshape(M, D)
    w2= wv[M * D:M * D + M2 * M] # 中間層ニューロンへの重み
    w2= w2.reshape(M2, M)
    v = wv[M * D + M2 * M:] # 出力層ニューロンへの重み
    v = v.reshape(K, M2)
    # ① x を入力して y を得る
    y, a, z, z2, b, b2, outb, outb2 = FNN(wv, M, M2, K, x, bn1, bn2,True)
    # 出力変数の準備
    dwv = np.zeros_like(wv)
    dw = np.zeros((M, D),dtype=np.float32)
    dw2= np.zeros((M2,M),dtype=np.float32)
    dv = np.zeros((K, M2),dtype=np.float32)

    delta2 = y - t
    delta12_1 = np.dot(delta2, v)
    delta12_2 = delta12_1 * (outb2 > 0.0)
    delta12_3 = bn2.backward(delta12_2)

    delta1_1 = np.dot(delta12_3, w2)
    delta1_2 = delta1_1 * (outb > 0.0)
    delta1_3 = bn1.backward(delta1_2)

    dv = np.dot(delta2.T, z2) / N
    dw2 = np.dot(delta12_3.T, z) / N
    dw = np.dot(delta1_3.T, x) / N

    # dw と dv を合体させて dwv とする
    dwv = np.c_[dw.reshape((1, M * D)), dw2.reshape((1, M2 * M)), dv.reshape((1,K * M2))]
    dwv = dwv.reshape(-1)
    return dwv

# 解析的微分を使った勾配法 -------
def Fit_FNN(wv_init,M,M2, K, x_train, t_train, x_test, t_test, stopval,bn1,bn2):
    N, D = x_train.shape  # 入力次元
    wv = wv_init.copy()
    err_train = np.float32(0)
    err_test = np.float32(0)
    alpha = 0.22

    A2 = 0.0127
    BatchSize = 256
    for i in range(2000):
        print("fit{0}".format(i))
        l = list(range(N))  # [0, 1, 2, 3, 4]
        random.shuffle(l)  # [4, 3, 2, 1, 0]

        x_train3 = x_train[l, :]
        t_train3 = t_train[l, :]

        for j in range(N//BatchSize):
            wv = wv - A2 * alpha * dCE_FNN(
                wv, M, M2, K,
                x_train3[BatchSize * j:min(BatchSize * j + BatchSize, N), :],
                t_train3[BatchSize * j:min(BatchSize * j + BatchSize, N), :],
                bn1,bn2
            )  # (A)
            bn1.modyfyGB(A2 * alpha)
            bn2.modyfyGB(A2 * alpha)

        err_train = CE_FNN(wv, M, M2, K, x_train, t_train, bn1, bn2, False)
        err_test = CE_FNN(wv, M, M2, K, x_test, t_test, bn1, bn2, False)
        print(err_train, err_test)
        alpha = np.maximum(np.minimum(alpha, err_train), np.float32(0.22))

        print(wv)
        print(np.max(wv), np.min(wv))
        print()
        print(np.max(bn1.gamma), np.min(bn1.gamma))
        print(np.max(bn1.beta), np.min(bn1.beta))
        print(np.max(bn1.running_mean), np.min(bn1.running_mean))
        print(np.max(bn1.running_var), np.min(bn1.running_var))
        print(np.max(bn1.xc), np.min(bn1.xc))
        print(np.max(bn1.std), np.min(bn1.std))
        print(np.max(bn1.dgamma), np.min(bn1.dgamma))
        print(np.max(bn1.dbeta), np.min(bn1.dbeta))
        print()
        print(np.max(bn2.gamma), np.min(bn2.gamma))
        print(np.max(bn2.beta), np.min(bn2.beta))
        print(np.max(bn2.running_mean), np.min(bn2.running_mean))
        print(np.max(bn2.running_var), np.min(bn2.running_var))
        print(np.max(bn2.xc), np.min(bn2.xc))
        print(np.max(bn2.std), np.min(bn2.std))
        print(np.max(bn2.dgamma), np.min(bn2.dgamma))
        print(np.max(bn2.dbeta), np.min(bn2.dbeta))

        if err_train < stopval:
            break


    return wv, err_train, err_test, bn1, bn2

def CreateWV(D,M,M2,K):
    wv0 = np.float32(np.random.randn(D * M) / np.sqrt(D / 2))
    wv1 = np.float32(np.random.randn(M2 * M) / np.sqrt(M / 2))
    wv2 = np.float32(np.random.randn(K * M2) / np.sqrt(M2 / 2))
    return np.hstack((wv0, wv1, wv2))
