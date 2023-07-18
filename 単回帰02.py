import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np

df1 = pd.read_csv('/content/drive/MyDrive/iris.csv',sep=',')

# (1)の解答
plt.scatter(df1['petal_length'],df1['petal_width'],color="red",alpha=0.5)
plt.xlabel("花びらの長さ（cm）")
plt.ylabel("花びらの幅（cm）")
plt.title("花びらの幅と長さの分散")
plt.show()


# (2)の解答
# 回帰分析をして傾きと切片を求める関数
def linear_regression(df):
    x = df['petal_length'].to_numpy()
    y = df['petal_width'].to_numpy()
    x_mean = df['petal_length'].mean()
    y_mean = df['petal_width'].mean()

    Sxy = np.cov([x, y], ddof=0)[0, 1]
    Sxx = np.var(x)
    a = Sxy / Sxx
    b = y_mean - a * x_mean
    return a, b

# 単回帰分析をした結果をa,bに入れる
a, b = linear_regression(df1)
# 小数点以下第3位まで表示
print(f'\n\n傾きの値 : {a:.3f}, 切片の値 : {b:.3f}\n\n')
# 分散をプロット
plt.scatter(df1['petal_length'],df1['petal_width'],color="red",alpha=0.5)
# 回帰直線を表示
plt.plot(df1['petal_length'],a*df1['petal_length']+b,color="black")
# タイトル・ラベルの設定
plt.xlabel("花びらの長さ（cm）")
plt.ylabel("花びらの幅（cm）")
plt.title("花びらの幅と長さの分散と回帰直線")
plt.show()