"""
1. ディレクトリの設定
2. データ拡張の設定
3. モデルの定義
4. K-fold交差検証の実行
5. 最終モデルの学習と保存
6. ROC曲線とPR曲線を使用した閾値の最適化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, auc

# ディレクトリの設定
train_dir = './data/augmented/'
original_train_dir = './data/train/'

# データ拡張の設定, モデルの汎化性能を向上させるため(色々試す場所)
datagen = ImageDataGenerator( # 画像データの前処理と拡張を行う
    rescale=1./255, # 全てのピクセル値を0から1の範囲に正規化
    rotation_range=40, # 画像をランダムに回転させる角度の範囲を指定(何でもいい。ここでは-40度から40度)
    width_shift_range=0.2, # 画像をランダムに横シフトさせる範囲(ここでは20%)
    height_shift_range=0.2, # 画像をランダムに縦シフトさせる範囲(ここでは20%)
    shear_range=0.2, # せん断変換の角度を指定、斜めにできる
    zoom_range=0.2, # ランダムにズームイン・ズームアウト(ここでは80%～120%になる)
    horizontal_flip=True, # ランダムに水平方向に反転
    vertical_flip=True, # ランダムに素直方向に反転
    fill_mode='nearest' # 新しく作られた画素をどのように埋めるかを指定('nearest'は最も近い画素の値)
)

# 画像サイズの取得(最初の画像ファイルを参照)
sample_img = plt.imread(os.path.join(original_train_dir, 'broken', os.listdir(os.path.join(original_train_dir, 'broken'))[0]))
# 画像の形状（高さ、幅、チャンネル数）の内、高さと幅のみ取得
img_height, img_width = sample_img.shape[:2]

# バッチサイズ(色々試す場所)
batch_size = 32

# モデルの構築（単純化と正則化含む）(色々試す場所)
def create_model():
    model = Sequential([ # Kerasの線形スタックモデル
        # 2次元の畳み込み層(フィルターの数(出力チャンネル数): 16, フィルターのサイズ: (3, 3), 活性化関数: ReLU, 入力画像のサイズとチャンネル数(3はRGB), L2正則化を適用して過学習を防ぐ)
        Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3), kernel_regularizer=l2(0.01)),
        MaxPooling2D(2, 2), # 最大プーリング層、   (2, 2)のウィンドウサイズで最大値を取る
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01)), # 2つ目の畳み込み層,フィルター数を32に増加
        MaxPooling2D(2, 2), # 2つ目の最大プーリング層
        Flatten(), # 多次元の入力を1次元に平坦化
        # 全結合層(ニューロン: 32個、活性化関数: ReLU)
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)), 
        Dropout(0.5), # 過学習を防ぐためにランダムに50%のニューロンを無効化
        Dense(1, activation='sigmoid') # 出力層(1つのニューロンを持ち、シグモイド活性化関数を使用(二値分類のため))
    ])

    # モデルをコンパイル
    model.compile(optimizer=Adam(learning_rate=0.0001), # 学習率0.0001のAdamオプティマイザー
                  loss='binary_crossentropy', # 二値分類のための損失関数
                  metrics=['accuracy']) # 評価指標として精度を使用
    return model

# K-fold交差検証の設定
n_splits = 5 # K-fold交差検証で使用するフォールド(分割)の数を5に設定
# フォールドの数を指定, shuffle=True: データをシャッフルしてから分割
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# コールバックの設定(色々試す場所)
# 過学習を防ぐために、検証損失が改善しなくなったら学習を早期に停止
# ここでは15エポック連続で改善が見られなければ学習を停止
# restore_best_weights: 監視対象の数量の最良の値を持つエポックからモデルの重みを復元する, 過学習の緩和
early_stopping = EarlyStopping(patience=15, min_delta=0.001, restore_best_weights=True)
# 学習率を動的に調整
# factor=0.2: 学習率を現在の値の20%に減少, 5エポック連続で改善が見られなければ学習率を減少
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001)

# データセットの構造を確認
print("トレーニングディレクトリの内容:")
print(os.listdir(train_dir))

# すべての画像ファイルのリストを作成
all_images = [] # 全ての画像ファイルのパスを格納するリスト
labels = [] # 各画像に対応するラベルを格納するリスト
class_names = os.listdir(train_dir) # 訓練データディレクトリ内のサブディレクトリ名（クラス名）のリスト
# 各クラスに対して
for class_name in class_names:
    class_path = os.path.join(train_dir, class_name)
    # 全ての画像の中の、各画像に対して
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)  # 画像ファイルの完全なパスを作成
        all_images.append(img_path) # 画像パスをリストに追加
        labels.append(class_names.index(class_name)) # クラスのインデックスをラベルとしてリストに追加

print(f"画像総数: {len(all_images)}") # 処理された画像の総数


# カスタムデータジェネレータ
# バッチ単位で画像データを生成
# file_list: 画像ファイルパスのリスト, batch_size: 一度に生成する画像の数, target_size: 画像のリサイズ先のサイズ
def custom_generator(file_list, batch_size, target_size):
    while True:
        # file_listからランダムにbatch_size個の画像パスを選択
        batch_paths = np.random.choice(file_list, batch_size)
        batch_images = []
        batch_labels = []
        # バッチ内の各画像に対して
        for path in batch_paths:
            img = load_img(path, target_size=target_size) # 指定されたパスから画像を読み込み、指定されたサイズにリサイズ
            img = img_to_array(img) # numpy配列に変換
            img = datagen.random_transform(img) # データ拡張を適用
            img = img / 255.0 # ピクセル値を0-1の範囲に正規化
            batch_images.append(img) # 処理された画像をバッチリストに追加
            # ファイルパスに'broken'が含まれていれば1（割米）、そうでなければ0（整流米）をラベルとする
            label = 1 if 'broken' in path else 0
            batch_labels.append(label) # ラベルをバッチラベルリストに追加
        # 処理されたバッチ画像とラベルをnumpy配列として返す(yieldで、メモリ効率の良いジェネレータにする)
        yield np.array(batch_images), np.array(batch_labels)


# K-fold交差検証の実行
histories = [] # 各フォールドの学習履歴を保存するための空リスト
fold_no = 1 # 現在のフォールド番号
n_epochs = 30  # エポック数を30に設定(色々試す場所)

# kfold.split(): データセットを訓練セットと検証セットに分割するためのインデックスを生成
# 各フォールドごとに
for train_index, val_index in kfold.split(all_images):
    print(f'Fold {fold_no}')

    # 訓練データと検証データのファイルリスト作成
    train_files = [all_images[i] for i in train_index]
    val_files = [all_images[i] for i in val_index]

    # 訓練データと検証データのジェネレータ作成(バッチ単位)
    train_generator = custom_generator(train_files, batch_size, (img_height, img_width))
    validation_generator = custom_generator(val_files, batch_size, (img_height, img_width))

    model = create_model() # 各フォールドで新しいモデルから学習を開始

    history = model.fit(
        train_generator,  # 訓練データのジェネレータ
        steps_per_epoch=len(train_files) // batch_size, # 1エポックあたりの訓練ステップ数
        validation_data=validation_generator, # 検証データのジェネレータ
        validation_steps=len(val_files) // batch_size, # 1エポックあたりの検証ステップ数
        epochs=n_epochs, # epoch数
        callbacks=[early_stopping, reduce_lr], # コールバック
        verbose=1 # 学習の進捗を表示
    )
    # 学習履歴を保存し、フォールド番号をインクリメント
    histories.append(history)
    fold_no += 1

# K-fold交差検証の結果の可視化
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
for i, history in enumerate(histories):
    plt.plot(history.history['accuracy'], label=f'Training Accuracy Fold {i+1}')
    plt.plot(history.history['val_accuracy'], label=f'Validation Accuracy Fold {i+1}')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
for i, history in enumerate(histories):
    plt.plot(history.history['loss'], label=f'Training Loss Fold {i+1}')
    plt.plot(history.history['val_loss'], label=f'Validation Loss Fold {i+1}')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('kfold_results.png') 
plt.show()

# 最終モデルの学習と保存
print("最終モデルの学習を開始")

# データを訓練用と検証用に分割
train_files, val_files = train_test_split(all_images, test_size=0.2, random_state=42)

train_generator = custom_generator(train_files, batch_size, (img_height, img_width))
validation_generator = custom_generator(val_files, batch_size, (img_height, img_width))

final_model = create_model()

# ModelCheckpointの設定(学習中にモデルの重みを保存するためのKerasコールバック)
checkpoint = ModelCheckpoint('best_rice_classification_model.keras', # モデルファイル名
                             monitor='val_loss', # 監視する指標。ここでは検証損失
                             mode='min', # 監視指標が最小なら最良のモデルと判断
                             save_best_only=True, # 最良のモデルのみを保存
                             verbose=1) # 保存時にメッセージを表示

# 最終モデル学習
final_history = final_model.fit(
    train_generator, # 訓練データを生成するジェネレータ
    steps_per_epoch=len(train_files) // batch_size, # 1エポックあたりの訓練ステップ数
    validation_data=validation_generator, # 検証データを生成するジェネレータ
    validation_steps=len(val_files) // batch_size, # 1エポックあたりの検証ステップ数
    epochs=n_epochs, # 学習を行うエポック数
    callbacks=[early_stopping, reduce_lr, checkpoint], # 使用するコールバックのリスト
    verbose=1 # 学習の進捗を表示
)

print("最終モデルの学習を終了")
print("ベストモデルを保存 'best_rice_classification_model.keras'")

# 最終モデルの学習曲線の表示
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(final_history.history['accuracy'], label='Training Accuracy')
plt.plot(final_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Final Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(final_history.history['loss'], label='Training Loss')
plt.plot(final_history.history['val_loss'], label='Validation Loss')
plt.title('Final Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('final_model_learning_curves.png') 
plt.show()

# 最終モデルの保存
final_model.save('rice_classification_model.keras')
print("念のため最終モデルを保存 'rice_classification_model.keras'")

# ROC曲線とPR曲線を使用した閾値の最適化
print("ROC曲線とPR曲線を使用した閾値の最適化を開始")

# モデルのロード
model = load_model('best_rice_classification_model.keras')

# 検証データの準備
validation_steps = len(val_files) // batch_size # 検証データのバッチ数を計算
y_true = [] # 真のラベル
y_pred = [] # 予測値

# バッチ数分ループ
for _ in range(validation_steps):
    x, y = next(validation_generator) # validation_generator から次のバッチのデータxとラベルyを取得
    y_true.extend(y) # 真のラベルyをy_trueリストに追加
    y_pred.extend(model.predict(x)) # モデルを使用して予測を行い、その結果をy_predリストに追加

# NumPy配列に変換
y_true = np.array(y_true)
y_pred = np.array(y_pred).flatten()

# ROC曲線の計算(fpr:偽陽性率, tpr:真陽性率, roc_thresholds:各点に対応する閾値)
fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr) # AUC

# ユーデンのJ統計量を最大化する閾値を見つける
j_scores = tpr - fpr
# J統計量が最大となる閾値を、最適なROC閾値として選択
best_threshold_roc = roc_thresholds[np.argmax(j_scores)]

# PR曲線の計算(precision: 適合率, recall: 再現率)
precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)

# F1スコアを最大化する閾値を見つける
f1_scores = 2 * (precision * recall) / (precision + recall) # 各閾値におけるF1スコアを計算
# F1スコアが最大となる閾値を、最適なPR閾値として選択
best_threshold_pr = pr_thresholds[np.argmax(f1_scores[:-1])]  # 最後の要素を除外(仕様)

# 結果の表示
print(f"ROC曲線に基づく最適な閾値: {best_threshold_roc:.4f}")
print(f"PR曲線に基づく最適な閾値: {best_threshold_pr:.4f}")

# ROC曲線のプロット
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# PR曲線のプロット
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='darkgreen', lw=2, label='PR curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.savefig('roc_pr_curves.png')
plt.show()

# 各閾値での性能評価(真のラベル, モデルの予想, 分類の決定に使用する閾値)
def eval_threshold(y_true, y_pred, threshold):
    # 予測値を二値分類結果に変換
    y_pred_bin = (y_pred > threshold).astype(int) # Trueを1にFalseを0に
    accuracy = np.mean(y_true == y_pred_bin) # 真と予測が一致している割合
    f1 = f1_score(y_true, y_pred_bin) # F1スコア
    return accuracy, f1 # 計算した精度とF1スコアを返す

# ROC曲線から得られた最適閾値を使って、精度とF1スコアを計算
roc_accuracy, roc_f1 = eval_threshold(y_true, y_pred, best_threshold_roc)
# PR曲線から得られた最適閾値を使って、精度とF1スコアを計算
pr_accuracy, pr_f1 = eval_threshold(y_true, y_pred, best_threshold_pr)

print(f"ROC閾値の性能 - 精度: {roc_accuracy:.4f}, F1スコア: {roc_f1:.4f}")
print(f"PR閾値の性能 - 精度: {pr_accuracy:.4f}, F1スコア: {pr_f1:.4f}")

print("閾値の最適化が完了しました。kometsubu.pyの閾値を更新してください。")