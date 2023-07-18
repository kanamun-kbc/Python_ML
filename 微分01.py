import sympy
import numpy as np
import math

###   (5)について

# シンボル
x = sympy.Symbol('x')
y = sympy.Symbol('y')
f = sympy.Symbol('f')

# 式本体
f = (x+y-5)**2 + (x*y-6)**2 + 4

# 各変数について偏微分
dx = sympy.diff(f, x)
dy = sympy.diff(f, y)

# listの形で極値となる(x,y)の値を格納
points = sympy.solve([dx, dy], [x, y])
print(points)
print("極値におけるxとyにおけるfの最小値を列挙する")
for i in range((int)(len(points)/2)):
  x_1,y_1 = points[i]
  print(f'x={x_1},y={y_1}においてのfの値は{f.subs([(x,x_1),(y,y_1)])}')



###    (6)について

# 多変量関数の微分
# 関数Eを最小にするx, y

# fは再装置を求めたい関数をdefで記述したPython関数
# 引数は成分をまとめたNumpy配列
# E(x,y)ではなくE(vec_w)
# vec_xのところは、Eの引数のNumpy配列と同じ次元のNumpy配列
# 勾配を求めたい点の座標(位置ベクトル)
def get_grad(f, vec_x, h=0.001):
  # 勾配ベクトルの初期化(vec_xと同じ形状のゼロ配列)
  grad = np.zeros_like(vec_x)
  for i in range(len(vec_x)):
    vec_i_org = vec_x[i]
    vec_x[i] = vec_i_org + h
    fh1 = f(vec_x)
    vec_x[i] = vec_i_org - h
    fh2 = f(vec_x)
    grad[i] = (fh1 - fh2) / (2*h)
    vec_x[i] = vec_i_org
  return grad
# ↑は中心差分近似をして、最終的な勾配ベクトルを返している

def E(vec_w):
    x = vec_w[0]
    y = vec_w[1]
    return (x**2)*(y**2)+x**2+y**2-10*x*y-10*x-10*y+65

for x in np.arange(0, 4.0, 0.001):
    for y in np.arange(0, 5.0, 0.001):
        grad = get_grad(E,[x,y],0.001)
        #print(grad)
        if abs(math.sqrt(grad[0]**2+grad[1]**2)) < 0.001:
            print(f'x={x},y={y}で最小になります。')
            break
