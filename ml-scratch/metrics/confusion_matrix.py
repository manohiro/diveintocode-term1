import numpy as np

class confusion_matrix:
    def accuracy_score(y_true, y_pred):
        """
        正解率(Accuracy)
        
        本来ポジティブに分類すべきデータをポジティブに分類したデータと、
        本来ネガティブに分類すべきデータをネガティブに分類できたデータの割合
        
        Parameters
        ----------
        y_ture : 次の形のndarray, shape (n_samples, 1)
            正解ラベルのデータ
        y : 次の形のndarray, shape (n_samples, 1)
            予測結果のラベルのデータ
        """
        TP = np.array((y_true == 1) & (y_pred == 1)).sum()
        FN = np.array((y_true == 1) & (y_pred == 0)).sum()
        FP = np.array((y_true == 0) & (y_pred == 1)).sum()
        TN = np.array((y_true == 0) & (y_pred == 0)).sum()
        
        return (TP + TN) / (TP + TN + FP + FN)
    
    def precision_score(y_true, y_pred):
        """
        精度(Precision)
        ポジティブに分類されたデータのうち、
        実際にポジティブであったデータの割合
        Parameters
        ----------
        y_ture : 次の形のndarray, shape (n_samples, 1)
            正解ラベルのデータ
        y : 次の形のndarray, shape (n_samples, 1)
            予測結果のラベルのデータ
        """
        TP = np.array((y_true == 1) & (y_pred == 1)).sum()
        FP = np.array((y_true == 0) & (y_pred == 1)).sum()
        
        return TP / (TP + FP)
    
    def recall_score(y_true, y_pred):
        """
        検出率(Recall)
        本来ポジティブに分類すべきデータを、
        正しくポジティブに分類できたデータの割合
        Parameters
        ----------
        y_ture : 次の形のndarray, shape (n_samples, 1)
            正解ラベルのデータ
        y : 次の形のndarray, shape (n_samples, 1)
            予測結果のラベルのデータ
        """
        TP = np.array((y_true == 1) & (y_pred == 1)).sum()
        FN = np.array((y_true == 1) & (y_pred == 0)).sum()
        
        return TP / (TP + FN)
    
    def f1_score(y_true, y_pred):
        """
        検出率(F1 Score)
        精度 (Precision) と検出率 (Recall) を
        バランス良く持ち合わせているかを示す指標
        Parameters
        ----------
        y_ture : 次の形のndarray, shape (n_samples, 1)
            正解ラベルのデータ
        y : 次の形のndarray, shape (n_samples, 1)
            予測結果のラベルのデータ
        """
        TP = np.array((y_true == 1) & (y_pred == 1)).sum()
        FN = np.array((y_true == 1) & (y_pred == 0)).sum()
        FP = np.array((y_true == 0) & (y_pred == 1)).sum()
        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN) 
        return 2 * (precision * recall) / (precision + recall)