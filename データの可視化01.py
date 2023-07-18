#演習14-01
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
#（１）の解答
df = pd.read_csv("/content/drive/MyDrive/pra14-01.csv")
plt.scatter(df["sepal_length"],df["sepal_width"], color='y')

plt.xlabel("sepalの長さ")
plt.ylabel("sepalの幅")
plt.title("あるアヤメのがくの長さと幅の関係")
plt.show()
#（２）の解答
print('sepalの長さの最大値は')
print(df.loc[:,['sepal_length']].max())
print('sepalの長さの最小値は')
print(df.loc[:,['sepal_length']].min())
#（３）の解答
plt.hist(df["sepal_length"], bins = 20, color = 'y', ec ='k')