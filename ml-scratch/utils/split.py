import numpy as np
import math

def train_test_split(X, y, train_size=0.8, stratify=False):
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
    
    
    if(stratify):
        # ラベルの種類を取得
        labels = np.unique(y_np)
        # 特徴変数の数
        feature_num = X_np.shape[1]
        # データ数
        data_num = len(y_np)
        # 結合
        data = np.hstack((X_np, y_np[:,None]))
        
        flag = False        
        # labelの数だけ回す
        for label in labels:            
            # ラベルごとにデータを抽出
            label_data = data[data[:,feature_num] == label]
            # 各ラベルごとのサイズ
            label_data_size = len(label_data)
            # tarinデータ数
            train_lines_num = math.floor(train_size * label_data_size)
            
            if(not flag):
                # 1回のみ
                flag = True
                # tarinデータ
                label_data_train = label_data[:train_lines_num:]
                # testデータ
                label_data_test = label_data[train_lines_num::]
            else:
                # 2回目以降 行列を結合
                # tarinデータ
                label_data_train = np.vstack((label_data_train, label_data[:train_lines_num:]))
                # testデータ
                label_data_test = np.vstack((label_data_test, label_data[train_lines_num::]))
                
        # xとyを分ける
        X_train = np.array(label_data_train)[:,:feature_num]
        y_train = np.array(label_data_train)[:,feature_num]
        
        X_test = np.array(label_data_test)[:,:feature_num]
        y_test = np.array(label_data_test)[:,feature_num]
    else:
        # 行数を取得
        X_np_size = len(X_np)
        y_np_size = len(y_np)

        # トレインサイズ
        X_tarin_size = math.floor(X_np_size * train_size)
        y_train_size = math.floor(y_np_size * train_size)
    
        # トレインデータ
        X_train = X_np[:X_tarin_size:]
        y_train = y_np[:y_train_size:]
    
        # テストデータ
        X_test = X_np[(X_tarin_size)::]
        y_test = y_np[(y_train_size)::]        

    return X_train, X_test, y_train, y_test