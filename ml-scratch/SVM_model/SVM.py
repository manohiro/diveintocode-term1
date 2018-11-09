class ScratchSVM:
    def __init__(self, num_iter, lr):
        self.iter = num_iter
        self.lr = lr
        self.y_label = []
        self.sv_lt = []
        self.a_sv = []        

    def fit(self, X, y):
        # ラグランジュ係数
        a = np.random.rand(len(X), 1)
        
        # 各ラベルのサポートベクターが見つかるまでループ
        for num in range(self.iter):
            # データ数分ループ
            for i in range(len(X)):
                temp = 0
                for j in range(len(X)):
                    temp += a[j]*y[i]*y[j]*np.dot(X[i].T, X[j])
                # ラグランジュ係数更新式
                a[i] = a[i] + self.lr * (1 - temp)
                # ラグランジュ係数がマイナスの場合
                if(a[i] < 0):
                    a[i] = 0
        # サポートベクター判定
        self.a_sv = a[a >= 1e-5]
        index_sv = np.where(a >= 1e-5)[0]
        self.y_label = y[index_sv]
        self.sv_lt = X[index_sv]       
        
    def predict(self, X):
        # 推定
        f_x = 0
        for i in range(len(self.sv_lt)):
                f_x = f_x + self.a_sv[i] * self.y_label[i] * np.dot(X, self.sv_lt[i].T)
        return f_x