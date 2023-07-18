import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

## --------------------------------準備-------------------------------
# 20000個のデータの読み込み
df1 = pd.read_csv('/content/drive/MyDrive/california_housing.csv')

# Priceを除いたdf1
X_df1 = df1.drop('Price',axis=1)
# それをnumpyに変換
X = X_df1.to_numpy()
# Priceのみ取り出したdf1をnumpyに変換
# 真の値Price1はこれのこと
Y = df1['Price'].to_numpy()



##-------------------------------------(1)---------------------------------

model = LinearRegression()
# 予想モデルを作成
model.fit(X,Y)
# 各値を入れていく(やらなくてもいい)
A = model.coef_[0]
B = model.coef_[1]
C = model.coef_[2]
D = model.coef_[3]
E = model.coef_[4]
F = model.coef_[5]
G = model.coef_[6]
H = model.coef_[7]
I = model.intercept_

print('(1)の解答')
print('予測式は以下のようになる\n')
print(f'Price ＝ {A} * MedInc + {B} * HouseAge + {C} * AveRooms \n+ {D} * AveBedrms +  {E} * Population + {F} * AveOccup\n+ {G} * Latitude + {H} * Longitude + {I}\n\n')




## -------------------------------------(2)-----------------------------------

# 予測値Price_predをpandasのSeries(DFの1カラム)で取得
Price_pred_df1 = A * df1['MedInc'] + B * df1['HouseAge'] + C * df1['AveRooms'] + D * df1['AveBedrms'] +  E * df1['Population'] + F * df1['AveOccup']+ G * df1['Latitude'] + H * df1['Longitude'] + I
# それをnumpyに変換
Price_pred1 = Price_pred_df1.to_numpy()
# 真の値Price1はY(numpy)のこと

# MSEの計算
MSE1 = np.mean((Y - Price_pred1) ** 2)

print('(2)の解答')
print(f'学習した予測モデルを用いて、\n学習に用いたデータを予想した値と実測値との平均二乗誤差は\n{MSE1}\nである。\n\n')




## ------------------------------------(3)-----------------------------------

# 残り640個のデータの読み込み
df2 = pd.read_csv('/content/drive/MyDrive/california_housing_test.csv')
# df2からPriceのみを取り出したnumpy
Price2 = df2['Price'].to_numpy()

# (2)と同様、予測値を計算しnumpyに変換、MSEの計算
Price_pred_df2 = A * df2['MedInc'] + B * df2['HouseAge'] + C * df2['AveRooms'] + D * df2['AveBedrms'] +  E * df2['Population'] + F * df2['AveOccup']+ G * df2['Latitude'] + H * df2['Longitude'] + I
Price_pred2 = Price_pred_df2.to_numpy()
MSE2 = np.mean((Price2 - Price_pred2) ** 2)

print('(3)の解答')
print(f'学習した予測モデルを用いて、\n学習に用いなかったデータを予想した値と実測値との平均二乗誤差は\n{MSE2}\nである。\n')