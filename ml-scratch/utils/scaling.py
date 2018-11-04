import numpy as np

class ScratchStandardScaler():
    """
    標準化する。
    """
    def __init__(self):
        self.std_value = np.zeros(1)
        self.mean_value = np.zeros(1)
    def fit(self, X):
        """
        標準化のために平均と標準偏差を計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        """
        # array型に変換
        X_np = np.array(X)
        # 列ごとの標準偏差
        self.std_value = np.std(X_np, axis=0)
        # 列ごとの平均
        self.mean_value = np.mean(X_np, axis=0)

    def transform(self, X):
        """
        fitで求めた値を使い標準化を行う。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          特徴量

        Returns
        ----------
        X_scaled : 次の形のndarray, shape (n_samples, n_features)
          標準化された特緒量
        """
        # array型に変換
        X_np = np.array(X)
                           
        X_scaled = (X_np - self.mean_value)/self.std_value
        
        return X_scaled