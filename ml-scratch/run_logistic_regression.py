import argparse
import pandas as pd
import numpy as np

# モジュールを読み込む
from linear_model.linear_regression import ScratchLinearRegression
from utils.split import train_test_split
from utils.scaling import ScratchStandardScaler

# コマンドライン引数の設定
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='linear regression')

parser.add_argument('--iter', default=5000, type=int,
                    help='number of iterations')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--alpha', default=1e-3, type=float,
                    help='regularization parameters')
parser.add_argument('--no_bias', action='store_true',
                    help='without bias')
parser.add_argument('--verbose', action='store_true',
                    help='output of learning process')
parser.add_argument('--dataset', default='train.csv', type=str,
                    help='path to csvfile')

def main():
    # コマンドライン引数の読み込み
    args = parser.parse_args()
    # データセットの読み込み
    iris_df = pd.read_csv(args.dataset)
    mappitng = {"Iris-versicolor":1, "Iris-virginica":0}
    train_df = train_df[(iris_df.Species == "Iris-versicolor") | (iris_df.Species == "Iris-virginica")]
    X = train_df.drop(["Id", "Species"], axis=1)
    y = iris_df["Species"].map(mappitng)

    # データ分割
    X_train, y_train, X_val, y_val = train_test_split(X, y, True)
    model = ScratchLogisticRegression(args.iter, args.lr, args.alpha, args.no_bias, args.verbose)
    model.fit(X_train, y_train, X_val, y_val)

    train_loss = model.train_loss
    val_loss = model.val_loss

    y_pred = model.predict(X_val)

if __name__ == '__main__':
    main()