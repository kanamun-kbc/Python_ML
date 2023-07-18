import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df1 = pd.read_csv('/content/drive/MyDrive/maha.csv', sep=',')
# １次元のデータをnumpyに変換
df1_np = df1['X'].to_numpy()


# (1)の解答
# 平均値 μ
mu = np.mean(df1_np)
# 標準偏差 Sx
Sx = np.std(df1_np)
# マハラノビス距離のリスト
mahala_list = abs(df1_np - mu)/Sx
# それを表示
print('(1)の解答')
print('全てのマハラノビス距離')
print(mahala_list)


# (2)の解答
# 四分位範囲を求める
q0, q25, q50, q75, q100 = np.percentile(mahala_list, q = [0, 25, 50, 75, 100], method = 'midpoint')
# 第一、第二、第三を表示
print(f'\n(2)の解答')
print(f' 25パーセンタイル(第１四分位数) = {q25}')
print(f' 50パーセンタイル(第２四分位数) = {q50}')
print(f' 75パーセンタイル(第３四分位数) = {q75}')


# (3)の解答
# IQRを求める
iqr = q75 - q25
# 下限及び上限を求める
lower_fence = q25 - 1.5 * iqr
upper_fence = q75 + 1.5 * iqr
# 下限を下回るか、上限を上回る値を選択する(リスト内包表記)
outlier = [x for x in mahala_list if x < lower_fence or upper_fence < x ]
# それの表示
print(f'\n(3)の解答')
print('外れ値のマハラノビス距離は以下の通り')
print(outlier)


# (4)の解答
print('\n(4)の解答')
print(f'有効なデータの数は{len(mahala_list)-len(outlier)}個である。')


# (5)の解答
# mahala_listのnumpyをDFに変換
mahala_list_df = pd.DataFrame(mahala_list)
print('\n(5)の解答')
print('マハラノビス距離の箱ひげ図')
sns.set(style='darkgrid')
sns.catplot(mahala_list_df, kind ='box', palette='pastel')
plt.show()