# 訓練データを"./data/image/train/broken", "./data/image/train/proper"に配置

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# ディレクトリの設定
broken_dir = './data/train/broken/'
proper_dir = './data/train/proper/'
augmented_broken_dir = './data/augmented/broken/'
augmented_proper_dir = './data/augmented/proper/'

# ディレクトリが存在しない場合は作成
os.makedirs(augmented_broken_dir, exist_ok=True)
os.makedirs(augmented_proper_dir, exist_ok=True)

# データ拡張の関数
def augment_image(img):
    # 回転(ここでは-30度から30度)
    img = img.rotate(np.random.randint(-30, 30))

    # フリップ(50%の確率で左右反転・上下反転)
    if np.random.rand() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.rand() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # 明るさ・コントラストの調整(0.8倍から1.2倍)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(np.random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(np.random.uniform(0.8, 1.2))

    # ガウシアンノイズの追加
    img_array = np.array(img)
    noise = np.random.normal(0, 5, img_array.shape)
    noisy_img_array = img_array + noise
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8) # 値を0-255に収め、unit8型に変換
    img = Image.fromarray(noisy_img_array)

    return img # 拡張された画像を返す

# データ拡張の実行(元画像のディレクトリ, 保存先のディレクトリ, 各クラスの目標枚数)
def augment_data(input_dirs, output_dirs, target_count):
    # 入力ディレクトリと出力ディレクトリを対応させてループ
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        original_count = len([f for f in os.listdir(input_dir) if f.endswith('.jpg')]) # jpgの数をカウント
        # 元データが目標値より多い場合(今回の場合はありえない)
        if original_count >= target_count:
            # ランダムにサンプリング
            files = np.random.choice([f for f in os.listdir(input_dir) if f.endswith('.jpg')], 
                                     target_count, replace=False)
            for file in files: # コピー
                img = Image.open(os.path.join(input_dir, file))
                img.save(os.path.join(output_dir, file))
        else:
            # 拡張が必要な場合
            augment_count = target_count - original_count # 生成する必要のある画像の数を計算
            files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')] # ファイル内の全てのjpgのリストを作成
            for i in range(augment_count):
                file = np.random.choice(files) # 拡張する画像をランダムに選ぶ
                img = Image.open(os.path.join(input_dir, file))
                aug_img = augment_image(img) # 拡張
                aug_img.save(os.path.join(output_dir, f'{os.path.splitext(file)[0]}_aug_{i}.jpg')) # 名前を新しくして保存
            # 元の画像もコピー
            for file in files:
                img = Image.open(os.path.join(input_dir, file))
                img.save(os.path.join(output_dir, file))

# 目標とする画像数（両クラスの合計数に応じて適切に設定）
target_count = 2000  # この場合は各クラス2000枚

# バランスの取れたデータ拡張の実行
augment_data([broken_dir, proper_dir], 
            [augmented_broken_dir, augmented_proper_dir], 
            target_count)

print("バランスの取れたデータ拡張が完了しました。")

# 拡張後のデータ数の確認
print(f"拡張後の割米画像数: {len(os.listdir(augmented_broken_dir))}")
print(f"拡張後の整流米画像数: {len(os.listdir(augmented_proper_dir))}")

# サンプル画像の表示(ランダムに5枚横一列に表示)
def show_sample(dir_path, num_samples=5):
    plt.figure(figsize=(15, 3))
    for i, filename in enumerate(np.random.choice(os.listdir(dir_path), num_samples, replace=False)):
        img = Image.open(os.path.join(dir_path, filename))
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(filename)
    plt.show()

print("拡張後の割米サンプル:")
show_sample(augmented_broken_dir)
print("拡張後の整流米サンプル:")
show_sample(augmented_proper_dir)