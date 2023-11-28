import cv2
import datetime
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# グローバル変数としてframeを宣言
frame = None

def save_image(frame, file_path):
    # ファイルを保存する
    cv2.imwrite(file_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print("画像を保存しました.")

def show_camera():
    global frame  # frameをグローバル変数として宣言
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        root.destroy()
        return
    # OpenCVのBGR形式をRGB形式に変換
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # フレームをPIL形式に変換
    img = Image.fromarray(frame)
    # PIL形式をtkinter PhotoImage形式に変換
    imgtk = ImageTk.PhotoImage(image=img)
    # カメラ映像を表示
    label.imgtk = imgtk
    label.config(image=imgtk)
    # カメラ映像の表示を更新
    label.after(10, show_camera)

def save_image_callback():
    global frame  # frameをグローバル変数として宣言
    if frame is not None:
        # ファイルの保存先を指定
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            save_image(frame, file_path)

# カメラのデバイス番号。通常は0か1です。
camera_index = 0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# GUIウィンドウを作成
root = tk.Tk()
root.title("Camera GUI")

# カメラ映像を表示するラベル
label = tk.Label(root)
label.pack()

# 保存ボタンを作成
save_button = tk.Button(root, text="保存", command=save_image_callback)
save_button.pack()

# カメラ映像の表示を開始
show_camera()

# GUIウィンドウを起動
root.mainloop()

# リソースを解放
cap.release()
cv2.destroyAllWindows()
