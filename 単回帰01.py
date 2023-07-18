import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np

df1 = pd.read_csv('/content/drive/MyDrive/virus_patient.csv',sep='\t')

# (1)の解答
# 分散をプロット
plt.scatter(df1['X'],df1['y'],c="red",alpha=0.5)
# タイトル・ラベルを設定
plt.title("新規感染者数と重症者数の散布図")
plt.xlabel("新規感染者数(万人)")
plt.ylabel("重症者数(万人)")
plt.grid()
plt.show()



# (2)の解答
# 単回帰分析をして傾きと切片を求める関数
def linear_regression(df):
    x = df['X'].to_numpy()
    y = df['y'].to_numpy()
    x_mean = df['X'].mean()
    y_mean = df['y'].mean()

    Sxy = np.cov([x, y], ddof=0)[0, 1]
    Sxx = np.var(x)
    a = Sxy / Sxx
    b = y_mean - a * x_mean
    return a, b

# 単回帰分析をした結果をa,bに入れる
a, b = linear_regression(df1)
# 小数点以下第3位まで表示
print(f'\n\n傾きの値 : {a:.3f}, 切片の値 : {b:.3f}\n\n')



# (3)の解答
# 回帰直線から見たい幅の値を入力
bind = (float)(input("予想される重傷者数が収まる範囲の予想値(万人)\n"))
print(f'回帰分析により、\n新規感染者数10万人から重傷者は{a*10+b:.3f}万人と予想される\n')
print(f'そこから、±{bind}万人の幅を取ったときの回帰直線を青色で表示する')
# 分散をプロット
plt.scatter(df1['X'],df1['y'],c="red",alpha=0.5)
# 回帰直線のx軸の範囲を設定
x = np.arange(0,12)
# 回帰直線を表示
plt.plot(x,a*x+b,color="black")
# 回帰直線±bindの直線を表示
plt.plot(x,a*x+b+bind,color="blue",alpha=0.5)
plt.plot(x,a*x+b-bind,color="blue",alpha=0.5)
# グラフの範囲設定とラベル・タイトルの設定
plt.xlim(0,10)
plt.ylim(0,4.0)
plt.title("新規感染者数と重症者数の散布と回帰直線")
plt.xlabel("新規感染者数(万人)")
plt.ylabel("重症者数(万人)")
plt.grid()
plt.show()