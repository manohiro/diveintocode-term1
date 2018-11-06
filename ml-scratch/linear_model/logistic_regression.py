import numpy as np # 線形代数ライブラリー
import matplotlib.pyplot as plt # 描画ライブラリー

class ScratchLogisticRegression:
    """
    線形回帰
    ＊コンストラクタ（__init__）のパラメータはここに書いておくと分かりやすい

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    alpha : float
      正則化パラメータ
    no_bias : bool
      バイアス項を入れない場合はTrue
    verbose : bool
      学習過程を出力する場合はTrue

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.train_loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録

    """

    def __init__(self, num_iter, lr, alpha, no_bias, verbose):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.alpha = alpha
        self.no_bias = no_bias
        self.verbose = verbose
        # 損失を記録する配列を用意
        self.train_loss = []
        self.val_loss = []
        # 学習した重み(パラメータ)
        self.__theta = np.zeros(1)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        ロジスティクス回帰を学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの正解値
        """
        # パラメータ推定
        self.__gradient_descent(X, y, X_val, y_val)
        
        if self.verbose:
            #verboseをTrueにした際は学習過程を出力
            print(np.array(self.train_loss))


    def predict(self, X):
        """
        ロジスティクス回帰を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            ロジスティクス回帰による推定結果
        """
        # 閾値で1, 0を分ける
        rs_label1 = self.__logistic_hypothesis(X) 
        rs_label1 = np.where(rs_label1[:,None] >= 0.5, 1, 0)
        rs_label0 = np.where(rs_label1 < 0.5, 1, 0)
        return np.hstack((rs_label0, rs_label1))

    def predict_proba(self, X):
        """
        ロジスティクス回帰を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            ロジスティクス回帰による推定結果
            をラベルごとの確率で出力
        """
        rs_label1 = self.__logistic_hypothesis(X)
        rs_label0 = 1 - rs_label1[:,None]
        return np.hstack((rs_label0, rs_label1[:,None]))
    
    def __gradient_descent(self, X, y, X_val, y_val):
        """
        最急降下法の計算
        
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples,)
          学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
          学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """
        # 損失を記録する配列を初期化
        self.train_loss = []
        self.val_loss = []
        
        # パラメータthetaを乱数シードで代入
        theta = np.random.randn(X.shape[1])
        m = len(X)
        
        # バイアス項判断
        if(not self.no_bias):
            theta_0 = np.random.randn(1)
        
        # iter分回す
        for i in range(self.iter):
            # 学習用
            if(self.no_bias):
                # 仮定関数 h_θ(x)
                h = 1 /(1 + np.exp(-1 * np.dot(X, theta)))
                # 目的関数(損失関数) J(θ)
                J = 1/m * np.sum(np.dot(-1 * y.T, np.log(h)) - np.dot(np.log(1-h), (1 - y ))) + self.alpha/2 * np.sum(theta)
            else:
                # バイアス項あり
                # 仮定関数 h_θ(x)
                h = 1 /(1 + np.exp(-1 * np.dot(np.insert(X, 0, 1, axis=1), np.insert(theta, 0, theta_0, axis=1))))
                # 目的関数(損失関数) J(θ)
                J = 1/m * np.sum(np.dot(-1 * y.T, np.log(h)) - np.dot(np.log(1-h), (1 - y ))) + self.alpha/2 * np.sum(np.insert(theta, 0, theta_0, axis=1))
                # バイアス項の更新式
                theta_0 = theta_0 - self.lr/m * np.sum((h - y)) 
            # 損失記録
            self.train_loss.append(J)
            # 更新式
            theta = theta - self.lr/m * np.sum(np.dot((h - y), X)) + self.alpha/m * theta

            # 検証用データ用
            if(X_val is not None and y_val is not None):
                if(self.no_bias):
                    # 仮定関数 h_θ(x)
                    h_val = 1 /(1 + np.exp(-1 * np.dot(X_val, theta)))
                    # 目的関数(損失関数) J(θ)
                    J_val = 1/m * np.sum(np.dot(-1 * y_val.T, np.log(h_val)) - np.dot(np.log(1-h_val), (1 - y_val))) + self.alpha/2 * np.sum(theta)
                else:
                    # バイアス項あり
                    # 仮定関数 h_θ(x)
                    h_val = 1 /(1 + np.exp(-1 * np.dot(np.dot(np.insert(X_val, 0, 1, axis=1), np.insert(theta, 0, theta_0, axis=1)))))
                    # 目的関数(損失関数) J(θ)
                    J_val = 1/m * np.sum(np.dot(-1 * y_val.T, np.log(h_val)) - np.dot(np.log(1-h_val), (1 - y_val))) + self.alpha/2 * np.sum(np.insert(theta, 0, theta_0, axis=1))
                # 損失記録
                self.val_loss.append(J_val)
        
        # 最終更新パラメータ
        if(self.no_bias): 
            self.__theta = theta
        else:
            self.__theta = np.insert(theta, 0, theta_0)
        
    def __logistic_hypothesis(self, X):
        """
        ロジスティクス回帰の仮定関数を計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ

        Returns
        -------
          次の形のndarray, shape (n_samples, 1)
          ロジスティクス回帰の仮定関数による推定結果

        """
        # バイアス項判断
        if(not self.no_bias):
            X = np.insert(X, 0, 1, axis=1)
        
        # 仮想関数
        # ラベル1の確率
        h = 1 /(1 + np.exp(-1 * np.dot(X, self.__theta)))

        return h
    
    def mode_loss(self):
        """
        学習曲線のプロット

        """
        x = list(range(1, (len(self.train_loss))+1))
        # 学習データ用
        plt.plot(x, np.array(self.train_loss), linewidth=5, label="train_loss")
        
        # 検証データ用
        if(self.val_loss):
            plt.plot(x, np.array(self.val_loss), linewidth=5, label="val_loss")
        
        plt.legend()
        plt.title('mode_loss')
        plt.xlabel('iter')
        plt.ylabel('loss')