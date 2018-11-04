import numpy as np
import math

def train_test_split(X, y, train_size=0.8,):
    """
    学習用データを分割する。

    Parameters
    ----------
    X : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    y : 次の形のndarray, shape (n_samples, )
      正解値
    train_size : float (0<train_size<1)
      何割をtrainとするか指定

    Returns
    ----------
    X_train : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    X_test : 次の形のndarray, shape (n_samples, n_features)
      検証データ
    y_train : 次の形のndarray, shape (n_samples, )
      学習データの正解値
    y_test : 次の形のndarray, shape (n_samples, )
      検証データの正解値
    """
    #ここにコードを書く
    # エラー処理
    if( train_size <=0 and train_size >=1):
        print("error:train_size is 0<train_size<1")
        return
    
    # エラー処理サイズチェック   
    X_np = np.array(X)
    y_np = np.array(y)
    
    # 行数を取得
    X_np_size = len(X_np)
    y_np_size = len(y_np)
    
    # トレインサイズ
    X_tarin_size = math.floor(X_np_size * train_size)
    y_train_size = math.floor(y_np_size * train_size)
    
    X_test_size = X_np_size - X_tarin_size
    y_test_size = y_np_size - y_train_size
    
    # トレインデータ
    X_train = X_np[:X_tarin_size]
    y_train = y_np[:y_train_size]
    
    # テストデータ
    X_test = X_np[(X_tarin_size + 1):]
    y_test = y_np[(y_train_size + 1):]

    return X_train, X_test, y_train, y_test