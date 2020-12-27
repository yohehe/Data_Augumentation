
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pathlib


'''
## 参考

quita\
https://qiita.com/BigTheAndo/items/4e8046998be9627ca85d

cv2
http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
'''

#path
data_path=pathlib.Path("/Users/kayanoyohei/Desktop/bird_test.jpg")

img=cv2.imread(str(data_path))

#cv2内で処理するため、RGBに変換しないほうが良い。
# ndarray(y,x,[B,G,R])を変形(y * x,[B,G,R])
Z = img.reshape((-1,3))
print(Z.shape)
# float32に型変換
Z = np.float32(Z)

#img出力
img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

"""
# K-means法
 K-means法とは、データのグループ分けを行うアルゴリズムです。このグループ分けのことを、機械学習分野ではクラスタリングと呼びます。

 今回グループ分けを行ったのは、画像の画素値です。カラー画像は１ピクセルごとに色を表す３つの数値(画素値)を持ち、この数値１つずつが青、緑、赤の画素値をあらわしています。これら３つの値を合わせて、ピクセル１つ１つの色を表現しています。

 処理の手順としては以下のとおりです。

ピクセルあたりの画素値を１つのデータとみなし、画像１枚分のデータをグループ分けします。
このグループ分けによってできたグループごとに、グループ内の平均値をとります。
グループに存在しているデータに、グループ内の平均値をわりあて、画像内の色の種類を減らします。
 このような処理を行うことで、画像内にたくさんあった色の種類が、グループ数と同じだけになります。


### cv2.kmeans パラメータ

入力パラメータ
samples : np.float32 型のデータとして与えられ，各特徴ベクトルは一列に保存されていなければなりません．

**nclusters(K)** : 最終的に必要とされるクラスタの数．

criteria : 繰り返し処理の終了条件です．この条件が満たされた時，アルゴリズムの繰り返し計算が終了します．実際は3個のパラメータのtuple ( type, max_iter, epsilon ) として与えられます:\

3.a - 終了条件のtype: 以下に示す3つのフラグを持っています:\

**cv2.TERM_CRITERIA_EPS** - 指定された精度(epsilon)に到達したら繰り返し計算を終了する． **cv2.TERM_CRITERIA_MAX_ITER** - 指定された繰り返し回数(max_iter)に到達したら繰り返し計算を終了する． **cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER** - 上記のどちらかの条件が満たされた時に繰り返し計算を終了する．

3.b - **max_iter** - 繰り返し計算の最大値を指定するための整数値．\

3.c - **epsilon** - 要求される精度．\
attempts : 異なる初期ラベリングを使ってアルゴリズムを実行する試行回数を表すフラグ．アルゴリズムは最良のコンパクトさをもたらすラベルを返します．このコンパクトさが出力として返されます．

flags : このフラグは重心の初期値を決める方法を指定します．普通は二つのフラグ cv2.KMEANS_PP_CENTERS と cv2.KMEANS_RANDOM_CENTERS が使われます．

"""
#k-meansの終了条件
# デフォルト値を使用

"""
まずはじめにcriteriaという変数で、K-means処理の終了条件を決定します。
K-means法にはイテレーション回数としきい値が存在しており、
それらをcriteriaで決めることができます。

K-Means関数を適用する前に criteria を指定する必要があります．
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)とすると、
ここでは繰り返し回数の上限を10回とし，精度が epsilon = 1.0 に達した時に終了するように終了条件を設定します．

"""
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 分割後のグループの数
K = 5
# k-means処理
_, label, center = cv2.kmeans(
                    Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
                    )
# float->np.uint8型に変換
center = np.uint8(center)
# グループごとにグループ内平均値を割り当て
res = center[label.flatten()]
# 元の画像サイズにもどす
res2 = res.reshape((img_rgb.shape))
print(res2.shape)
#>>>(600, 800, 3)
#K=5の場合は５色で描画された画像が出力される。
img_rgb=cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()




