# 分類したいデータはこのファイルから見て"./data/img"ディレクトリ内に入れて実行してください
# 訓練で使ったデータ全てを用いて評価をしたい場合は、"./data/train/broken","./data/train/proper"にそれぞれ訓練データを配置してください
# model_training.ipynbで生成したbest_rice_classification_model.kerasをダウンロードし、このファイルと同じ階層に配置してください
# 分類のみしたい場合、「python kometsubu.py ./data/img」
# 分類+訓練データでの評価を確認したい場合、「python kometsubu.py .data/img ./data/train」

import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score, classification_report

# model_trainingファイルで求めた閾値をここに入力
THRESHOLD = 0.5339
## -----画像の前処理の関数-----
# img_path: 処理する画像ファイルのパス
# target_size: 画像をリサイズするための目標サイズ
def preprocess_image(img_path, target_size):
    # 画像を読み込み、リサイズ
    img = load_img(img_path, target_size=target_size)
    # NumPy配列に変換
    img_array = img_to_array(img)
    # 配列に新しい軸（次元）を追加、モデルが期待する入力形状（バッチ処理のため）に合わせるため
    img_array = np.expand_dims(img_array, axis=0)
    # ピクセル値を0から1の範囲に正規化
    img_array /= 255.0
    # 前処理された画像データ（NumPy 配列）を返す
    return img_array


## -----前処理から分類までのまとめ関数-----
# model: 学習済みの機械学習モデル
# img_path: 分類する画像ファイルのパス
def classify_image(model, img_path):
    # モデルの入力テンソルの形状を取得(モデルが期待する入力画像のサイズ（高さと幅）を取得)
    input_shape = model.input_shape[1:3]
    # 画像前処理
    img_array = preprocess_image(img_path, input_shape)
    # モデルを使用して入力画像の分類(結果は各クラスに対する確率値の配列)
    prediction = model.predict(img_array)
    # 予測値が THRESHOLD より大きい場合は割米、そうでないなら整流米
    return "broken" if prediction[0][0] > THRESHOLD else "proper"


# -----訓練用に使ったデータでモデルの評価を行う関数(未知のデータでは使いません。あくまで訓練用での評価)-----
# model: 評価する学習済みの機械学習モデル, data_dir: 訓練データが格納されているディレクトリのパス
# ※ broken'と'proper'の2つのサブディレクトリを持つことを前提としている
def evaluate_model(model, data_dir):
    true_labels = [] # 実際のラベル（正解）を格納するためのリスト
    predict_labels = [] # モデルによる予測結果を格納するためのリスト
    
    # 'broken'（割米）と'proper'（整流米）のサブディレクトリへのパスを作成
    broken_dir = os.path.join(data_dir, 'broken')
    proper_dir = os.path.join(data_dir, 'proper')
    
    # directory: 各クラスの画像が格納されているディレクトリのパス
    # label: そのディレクトリ内の画像の正解ラベル
    for directory, label in [(broken_dir, "broken"), (proper_dir, "proper")]:

        # ディレクトリが存在しない場合、警告を出力してそのクラスの処理をスキップ
        if not os.path.isdir(directory):
            print(f"警告: ディレクトリ名: '{directory}' が存在しません")
            continue
        
        print(f"Processing class: {label}")
        # listdir()で指定されたディレクトリ内のファイル名のリストを取得
        for filename in os.listdir(directory):
            # 画像ファイルであることを確認
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(directory, filename)
                # 画像を分類
                result = classify_image(model, img_path)
                # 実際のラベル（label）と予測結果（result）をそれぞれのリストに追加
                true_labels.append(label)
                predict_labels.append(result)
    
    # 正解率の計算
    accuracy = accuracy_score(true_labels, predict_labels)
    # 分類の詳細なレポート（精度、再現率、F1スコアなどなど）を生成
    report = classification_report(true_labels, predict_labels)
    # 正解率と分類の詳細なレポートを返す
    return accuracy, report

# -----main関数-----
# img_dir: 分類する画像が含まれるディレクトリ(分類したいデータ(場所は"./data/img"))
# train_data_dir: オプションの引数。モデルを評価するための訓練データのディレクトリ
def main(img_dir, train_data_dir=None):
    # 分類したい画像データがなければプログラム終了
    if not os.path.exists(img_dir):
        print(f"警告: ディレクトリ名: '{img_dir}' が存在しません")
        sys.exit(1)

    # モデルファイルのパスを指定
    model_path = 'best_rice_classification_model.keras'
    # モデルがなければプログラム終了
    if not os.path.exists(model_path):
        print(f"警告: モデルファイル名: '{model_path}' が存在しません")
        sys.exit(1)

    # モデルをロード
    model = load_model(model_path)
    print(f"モデルのロードに成功しました。モデル名: {model_path}")
    print(f"モデルの入力形状: {model.input_shape}")

    # 指定されたディレクトリ内の全ての画像を分類(全ての画像に大してclassify_image関数を通す)
    print("\n指定されたディレクトリ内の全ての画像を分類:")
    for filename in os.listdir(img_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(img_dir, filename)
            result = classify_image(model, img_path)
            print(f"{filename} : {result}")

    # 訓練データを使用した評価
    if train_data_dir:
        if not os.path.exists(train_data_dir):
            print(f"警告: 訓練データのディレクトリ '{train_data_dir}' が存在しません")
            sys.exit(1)
        
        print("\n訓練データでのモデル評価:")
        accuracy, report = evaluate_model(model, train_data_dir)
        print(f"正解率: {accuracy:.4f}") # 正解率
        print("分類レポート:") # 詳細なレポート
        print(report)


if __name__ == "__main__":
    # コマンドライン引数の数をチェック
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python kometsubu.py <image_directory> [<training_data_directory>]")
        sys.exit(1)
    
    img_dir = sys.argv[1] # 分類したいデータのディレクトリ
    train_data_dir = sys.argv[2] if len(sys.argv) == 3 else None # 評価で使いたい訓練データのディレクトリ
    main(img_dir, train_data_dir)