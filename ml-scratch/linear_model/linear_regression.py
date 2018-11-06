import numpy as np # 線形代数ライブラリー
import matplotlib.pyplot as plt # 描画ライブラリー

class ScratchLinearRegression():
    """
    線形回帰
    ＊コンストラクタ（__init__）のパラメータはここに書いておくと分かりやすい

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
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

    def __init__(self, num_iter, lr, no_bias, verbose):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        # 損失を記録する配列を用意
        self.train_loss = []
        self.val_loss = []
        # 学習した重み(パラメータ)
        self.theta = np.zeros(1)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        線形回帰を学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """
        # パラメータ推定
        self.__gradient_descent(X, y, X_val, y_val)
        
        if self.verbose:
            #verboseをTrueにした際は学習過程を出力
            print(np.array(self.train_loss))


    def predict(self, X):
        """
        線形回帰を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            線形回帰による推定結果
        """

        return self._linear_hypothesis(X)
    
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
        
        # バイアス項判断
        if(self.no_bias):
            # パラメータthetaを乱数シードで代入
            theta = np.random.randn(X.shape[1])
        else:
            theta = np.random.randn((X.shape[1] + 1))
            X = np.insert(X, 0, 1, axis=1)
            X_val = np.insert(X_val, 0, 1, axis=1)
    
        # iter分回す
        for i in range(self.iter):
            # 学習用データ
            # 予測値
            y_hat = np.dot(X, theta)
            # 誤差
            E = y_hat - y
            # パラメータ更新
            theta = theta - self.lr/len(X) * np.sum(np.dot(E.T, X))
            # ロス結果追加
            self.train_loss.append(np.sum(E ** 2) / (2*len(y)))
            
            # 検証用データ用
            if(X_val is not None and y_val is not None):
                # 予測値
                y_hat_val = np.dot(X_val, theta)
                # 誤差
                E_val = y_hat_val - y_val
                # パラメータ更新
                theta = theta - self.lr/len(X_val) * np.sum(np.dot(E_val.T, X_val))
                # ロス結果追加
                self.val_loss.append(np.sum(E_val ** 2) / (2*len(y_val)))
            
        self.__theta = theta
        
    def __linear_hypothesis(self, X):
        """
        線形の仮定関数を計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ

        Returns
        -------
          次の形のndarray, shape (n_samples, 1)
          線形の仮定関数による推定結果

        """
        # バイアス項判断
        if(not self.no_bias):
            X = np.insert(X, 0, 1, axis=1)
        return np.dot(np.array(X), self.__theta)
    
    def mode_loss(self):
        """
        学習曲線のプロット

        """
        x = list(range(1, (len(self.train_loss))+1))
        plt.plot(x, np.array(self.train_loss), linewidth=10, label="train_loss")
        if(self.val_loss):
            plt.plot(x, np.array(self.val_loss), linewidth=10, label="val_loss")
        plt.legend()
        plt.title('mode_loss')
        plt.xlabel('iter')
        plt.ylabel('loss')