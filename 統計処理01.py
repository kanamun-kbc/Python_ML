import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np

df1 = pd.read_csv('/content/drive/MyDrive/data1.tsv', sep='\t')
#（１）の解答
plt.title("Xとｙの分散")
plt.xlabel("Xの値")
plt.ylabel("ｙの値")
plt.scatter(df1["X"],df1["ｙ"])
plt.show()
#（２）の解答
ary1 = df1.values
bunsany = np.var(ary1[:,1])
print(f'\nｙの分散は{bunsany}\n')
#（３）の解答
kyobunxy = np.cov(ary1[:,0],ary1[:,1],ddof=0)[0,1]
print(f'Xとｙの共分散は{kyobunxy}\n')
soukanxy = np.corrcoef(ary1[:,0],ary1[:,1])[0,1]
print(f'Xとｙの相関係数は{soukanxy}\n')
#（４）の解答
mse = np.mean((ary1[:,1]-ary1[:,2])**2)
print(f'ｙとy_predの平均二乗誤差は{mse}\n')
#（５）の解答
mae = np.mean(np.abs(ary1[:,1]-ary1[:,2]))
print(f'ｙとy_predの平均絶対値誤差は{mae}')