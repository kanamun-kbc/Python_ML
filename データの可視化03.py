import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np
import seaborn as sns

df1 = pd.read_csv('/content/drive/MyDrive/wine.data.csv', encoding='sjis')

#（１）の解答
mean_Aoa = df1['Alcalinity_of_ash'].mean()
print(f'全ワインのアルカリ灰の平均値{mean_Aoa}')

#（２）の解答
df_wine1 = df1[df1['class'].isin([1])]
mean_Aoa1 = df_wine1['Alcalinity_of_ash'].mean()
print(f'ワイン１のアルカリ灰の平均値{mean_Aoa1}')

#（３）の解答
df_class = df1.groupby('class').mean()
print(f'成分の平均値(ワインの種類ごと){df_class}')

#(４)の解答
plt.scatter(data=df1, x='Malic_acid',y='Alcohol')
plt.title('アルコールとリンゴ酸の散布図')
plt.show()

#（５）の解答
sns.scatterplot(data=df1,x='Malic_acid',y='Alcohol',hue='class',palette='pastel')
plt.title('アルコールとリンゴ酸の散布図(クラス別)')
plt.show()

#（６）の解答
sns.pairplot(data=df1[['Alcohol','Malic_acid','Ash','class']],hue='class',diag_kind='hist',palette='pastel')
plt.show()