import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np

df1 = pd.read_csv('c:\\Users\\Kanamu-Suehiro\\Desktop\\機械学習\\第１４回演習\\cityA-temps.csv', encoding='sjis')
#（１）の解答
a = []
b = []
for i in range(118):
  for j in range(12):
    a.append(1900+i)
    b.append(1+j)

array_a = np.array(a)
array_b = np.array(b)
dfa = pd.DataFrame(array_a)
dfb = pd.DataFrame(array_b)
df1['西暦年'] = dfa
df1['月'] = dfb
print(df1)
#（２）の解答
kaishinen = (int)(input('開始年を入力'))
tsuki = (int)(input('表示したい月を入力'))
#月の指定
df2 = df1[df1['月'].isin([tsuki])]
#開始年から下の行を取り出す
df3 = df2[df2.西暦年 >= kaishinen]
df3.plot(x='西暦年',y='A市の平均気温(℃)',color='y')
plt.title(f'A市の{tsuki}月の平均気温の長期推移')
plt.show()