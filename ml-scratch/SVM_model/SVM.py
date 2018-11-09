import numpy as np

class ScratchSVM:
    def __init__(self, num_iter, lr):
        self.iter = num_iter
        self.lr = lr
        self.y_label = []
        self.sv_lt = []
        self.a_sv = []        

    def fit(self, X, y):
        # ラグランジュ係数
        a = np.random.randn(len(X), 1)
        # イテレータの回数分回す
        for num in range(self.iter):
            flag1 = False # サポートベクタラベル1用 検出フラグ
            flag0 = False # サポートベクタラベル-1用 検出フラグ
            # データ数分ループ
            for i in range(len(X)):
                temp = 0
                for j in range(len(X)):
                    temp += a[j]*y[i]*y[j]*np.dot(X[i].T, X[j])
                # ラグランジュ係数更新式
                a[i] = a[i] + self.lr * (1 - temp)
                # サポートベクター判定
                if(a[i] >= 1e-5):
                    if( y[i] == 1):
                        if(not flag1):
                            # ラベル1用サポートベクター
                            sv_lt_1 = X[i]    # ラベル1のサポートベクター
                            a_sv_1 = a[i]     # ラベル1のラグランジュ係数
                            flag1 = True
                        else:
                            sv_lt_1 = np.vstack((sv_lt_1, X[i])) # ラベル1のサポートベクター
                            a_sv_1 = np.vstack((a_sv_1, a[i]))   # ラベル1のラグランジュ係数
                    else:
                        if(not flag0):
                            # ラベル-1用サポートベクター
                            sv_lt_0 = X[i]     # ラベル-1のサポートベクター
                            a_sv_0 = a[i]      # ラベル-1のラグランジュ係数
                            flag0 = True  
                        else:
                            sv_lt_0 = np.vstack((sv_lt_0, X[i])) # ラベル-1のサポートベクター
                            a_sv_0 = np.vstack((a_sv_0, a[i]))   # ラベル-1のラグランジュ係数
                # ラグランジュ係数がマイナスの場合
                if(a[i] < 0):
                    a[i] = 0
            # 各ラベルのサポートベクター取得判定
            if(flag1 and flag0):
                # ラベル
                self.y_label = np.vstack((np.full((len(sv_lt_1), 1),1), np.full((len(sv_lt_0),1) , -1)))
                # サポートベクター
                self.sv_lt = np.vstack((sv_lt_1, sv_lt_0))
                # ラグランジュ係数
                self.a_sv = np.vstack((a_sv_1, a_sv_0))
                
    def predict(self, X):
        # 推定
        f_x = 0
        for i in range(len(self.sv_lt)):
            f_x = f_x + self.a_sv[i] * self.y_label[i] * np.dot(X, self.sv_lt[i].T)
        return f_x