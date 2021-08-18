import time
import numpy as np
import random
from PIL import Image
import os
import mlp
np.set_printoptions(suppress=True)#強制的に小数表記

SLLLS = ["ＫＯ術", "アイテム使用強化", "ガード強化", "ガード性能", "キノコ大好き", "ジャンプ鉄人", "スタミナ急速回復", "スタミナ奪取", "ひるみ軽減", "フルチャージ", "ブレ抑制",
         "ボマー", "ランナー", "火事場力", "火属性攻撃強化", "火耐性", "会心撃【属性】", "回避距離ＵＰ", "回避性能", "回復速度", "滑走強化", "貫通弾・貫通矢強化", "気絶耐性",
         "鬼火纏", "逆恨み", "逆襲", "強化持続", "業物", "見切り", "幸運", "広域化", "攻めの守勢", "攻撃", "高速変形", "剛刃研磨", "散弾・拡散矢強化", "死中に活",
         "耳栓", "弱点特効", "集中", "匠", "乗り名人", "植生学", "心眼", "水属性攻撃強化", "水耐性", "睡眠属性強化", "睡眠耐性", "精霊の加護", "早食い", "装填拡張",
         "装填速度", "速射強化", "属性やられ耐性", "体術", "体力回復量ＵＰ", "耐震", "弾丸節約", "弾導強化", "地質学", "挑戦者", "超会心", "通常弾・連射矢強化", "泥雪耐性",
         "笛吹き名人", "砥石使用高速化", "特殊射撃強化", "毒属性強化", "毒耐性", "鈍器使い", "納刀術", "破壊王", "剥ぎ取り鉄人", "爆破やられ耐性", "爆破属性強化",
         "抜刀術【技】", "抜刀術【力】", "反動軽減", "飛び込み", "氷属性攻撃強化", "氷耐性", "不屈", "風圧耐性", "腹減り耐性", "壁面移動", "泡沫の舞", "砲術", "砲弾装填",
         "防御", "麻痺属性強化", "麻痺耐性", "満足感", "陽動", "雷属性攻撃強化", "雷耐性", "龍属性攻撃強化", "龍耐性", "力の解放", "渾身", "翔蟲使い", "達人芸",
         "NULL"]


def Fload(X,T,btc,K):
    s5 = os.getcwd()
    inpath = "H:\\mhrchap\\triming2"

    n = 0
    si = 0
    ei = 0
    for fldr in SLLLS:
        os.chdir(inpath)
        os.chdir(fldr)
        print(fldr)
        filenono = 0
        if os.path.isfile("{0}.npy".format(filenono)):
            pass
        else:
            filenono = 0

        sdata = np.load("{0}.npy".format(filenono))  # 65536,2816
        l = list(range(sdata.shape[0]))  # [0, 1, 2, 3, 4]
        random.shuffle(l)  # [4, 3, 2, 1, 0]
        ssdata = sdata[l, :]

        csz = btc
        if csz > sdata.shape[0]:
            csz = sdata.shape[0]
        ei += csz
        X[si:ei, :] = np.float32(ssdata[0:csz, :] / 255.0)
        T[si:ei, n] = 1.0
        si = ei
        n += 1
        if n==K:
            break

    print(ei)
    os.chdir(s5)
    return

def Seitoritu(T_test,y):
    N_test = T_test.shape[0]
    K = T_test.shape[1]
    anst = 0
    ansf = 0
    for i in range(N_test):
        sc = 0.0
        maxidx = -1
        sans = -1
        for j in range(K):
            if T_test[i, j] == 1:
                sans = j
            if y[i, j] > sc:
                sc = y[i, j]
                maxidx = j
        if T_test[i, maxidx] == 1:
            anst += 1
        else:
            ansf += 1
            print(SLLLS[sans],SLLLS[maxidx])
    return anst, ansf

def XtoXtrain_Xtest(X,T,testRatio):
    N = X.shape[0]
    l = list(range(N))  # [0, 1, 2, 3, 4]
    random.shuffle(l)  # [4, 3, 2, 1, 0]
    X_ = X[l, :]
    T_ = T[l, :]

    # N = 504955+1396  # データの数
    N_test = int(N / 1.0 * testRatio)
    N_training = N - N_test

    X_train = X_[:N_training, :]
    X_test = X_[N_training:, :]
    T_train = T_[:N_training, :]
    T_test = T_[N_training:, :]
    return X_train,X_test,T_train,T_test


def XtoXtrain_Xtest2(X,T):
    N = X.shape[0]
    l = list(range(N))  # [0, 1, 2, 3, 4]
    random.shuffle(l)  # [4, 3, 2, 1, 0]
    X_ = X[l, :]
    T_ = T[l, :]
    X_train = X_
    X_test = X_[0:256, :]
    T_train = T_
    T_test = T_[0:256, :]
    return X_train,X_test,T_train,T_test



def defload(bn1,bn2):
    #途中データロード
    WV = np.load("WV0.npy")

    bn1.gamma=np.load("bn1_gamma.npy")
    bn1.beta=np.load("bn1_beta.npy")
    bn1.momentum=np.load("bn1_momentum.npy")
    bn1.running_mean=np.load("bn1_running_mean.npy")
    bn1.running_var=np.load("bn1_running_var.npy")
    bn1.batch_size=np.load("bn1_batch_size.npy")
    bn1.xc=np.load("bn1_xc.npy")
    bn1.std=np.load("bn1_std.npy")
    bn1.dgamma=np.load("bn1_dgamma.npy")
    bn1.dbeta=np.load("bn1_dbeta.npy")

    bn2.gamma=np.load("bn2_gamma.npy")
    bn2.beta=np.load("bn2_beta.npy")
    bn2.momentum=np.load("bn2_momentum.npy")
    bn2.running_mean=np.load("bn2_running_mean.npy")
    bn2.running_var=np.load("bn2_running_var.npy")
    bn2.batch_size=np.load("bn2_batch_size.npy")
    bn2.xc=np.load("bn2_xc.npy")
    bn2.std=np.load("bn2_std.npy")
    bn2.dgamma=np.load("bn2_dgamma.npy")
    bn2.dbeta=np.load("bn2_dbeta.npy")
    return WV
# データ生成 ------------------------------------------------------------------------------------------------------------
# データ生成 ------------------------------------------------------------------------------------------------------------
np.random.seed(seed=124897356)  # 乱数を固定
#N = 415676  # データの数
M = 512 # 1層
M2 = 256 # 2層
D = 128*22 # 入力層
K = 102


"""
T = np.zeros((N, K), dtype=np.float32)
X = np.zeros((N, D) , dtype=np.float32)
Fload(X,T,4096,K)
print(X.dtype)
print(T.dtype)
X_train,X_test,T_train,T_test = XtoXtrain_Xtest(X,T,0.02)

X = 0
T = 0

np.save("X_train",X_train)
np.save("X_test",X_test)
np.save("T_train",T_train)
np.save("T_test",T_test)
exit(0)
"""

for iii in range(1867):
    zdata = np.load("H:\\mhrchap\\triming3\\{0}.npz".format(iii))
    X = zdata["x"]
    X = np.float32(X[:, :] / 255.0)
    N = X.shape[0]
    T = np.zeros((N, K), dtype=np.float32)
    zt = zdata["t"]
    for i in range(N):
        T[i, zt[i]] = np.float32(1.0)
    X_train, X_test, T_train, T_test = XtoXtrain_Xtest2(X, T)
    # メイン ---------------------------
    #WV_init = mlp.CreateWV(D,M,M2,K)
    bn1 = mlp.BatchNormalization(gamma=1, beta=0)
    bn2 = mlp.BatchNormalization(gamma=1, beta=0)
    WV_init = defload(bn1, bn2)
    WV, Err_train, Err_test, bn1, bn2 = mlp.Fit_FNN(WV_init, M, M2, K, X_train, T_train, X_test, T_test, 0.005996, bn1, bn2)
    if iii % 6 == 5:
        y, _, _, _, _, _, _, _ = mlp.FNN(WV, M, M2, K, X_test, bn1, bn2, False)
        anst, ansf = Seitoritu(T_test, y)
        print(anst, ansf)
        print("正答率{0}".format(anst / (anst + ansf)))
        if Err_test < 0.0062:
            np.save("WV0", WV)
            np.save("bn1_gamma", bn1.gamma)
            np.save("bn1_beta", bn1.beta)
            np.save("bn1_momentum", bn1.momentum)
            np.save("bn1_running_mean", bn1.running_mean)
            np.save("bn1_running_var", bn1.running_var)
            np.save("bn1_batch_size", bn1.batch_size)
            np.save("bn1_xc", bn1.xc)
            np.save("bn1_std", bn1.std)
            np.save("bn1_dgamma", bn1.dgamma)
            np.save("bn1_dbeta", bn1.dbeta)
            np.save("bn2_gamma", bn2.gamma)
            np.save("bn2_beta", bn2.beta)
            np.save("bn2_momentum", bn2.momentum)
            np.save("bn2_running_mean", bn2.running_mean)
            np.save("bn2_running_var", bn2.running_var)
            np.save("bn2_batch_size", bn2.batch_size)
            np.save("bn2_xc", bn2.xc)
            np.save("bn2_std", bn2.std)
            np.save("bn2_dgamma", bn2.dgamma)
            np.save("bn2_dbeta", bn2.dbeta)