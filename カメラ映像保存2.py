import cv2
import threading
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

# プログラムの設定やパラメータの管理
class Config():
    def __init__(self):
        # 録画開始のフラグ
        self.start_flag = 0
        # 録画停止のフラグ
        self.stop_flag = 0
        # カメラのフレームレート
        self.fps = 30
        # 録画時間の上限(秒)
        self.m_time = 10
        # カメラ映像の幅と高さ
        self.w, self.h = 640, 480
        # カメラキャプチャのためのオブジェクト
        self.cap = cv2.VideoCapture(0)

# カメラ映像の録画
class RecordThread():
    def __init__(self, config):
    	# configオブジェクトを受け取り、self.configに保存
        self.config = config

	# カメラ映像の録画
    def loop(self):
    	# 録画されたフレームが格納されるリスト
        self.config.img_list = []
        # 録画停止のフラグが1になるまで続ける
        while self.config.stop_flag != 1:
        	# 指定したフレームに合わせて一定の待ち時間を設ける
            time.sleep(1 / self.config.fps)
            # カメラから1フレームを取得
            # (retはブール型で正常に取得できたかどうか、frameはretがTrueのときの画像のNumpy配列)
            ret, frame = self.config.cap.read()
            # フレームが正常に取得されたら
            if ret:
            	# リストに格納
                self.config.img_list.append(frame)
                # リアルタイムで画像表示
                cv2.imshow('Recording', frame)
                # 「q」キーが押されたらループを終了
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

		# ループを抜けるとウィンドウを閉じる
        cv2.destroyAllWindows()

# GUIアプリケーションの構築
class CaptureApp(tk.Tk):
	# 初期化
    def __init__(self):
        super().__init__()
        # Configクラスのインスタンス生成
        self.config = Config()
        # GUIのレイアウト設定
        self.create_widgets()

	# GUIのレイアウト設定
    def create_widgets(self):
    	# レイアウト
        ttk.Label(self, text="Camera Capture App", font=("Helvetica", 16)).pack(pady=10)
        # ファイル名入力フィールド
        self.filename_entry = ttk.Entry(self)
        self.filename_entry.pack(pady=5)
		# ボタン(イベントに対応するメソッドを設定)
        self.start_button = ttk.Button(self, text="Start", command=self.record_start)
        self.stop_button = ttk.Button(self, text="Stop", command=self.record_stop)
        self.label = ttk.Label(self, text="ファイル名の指定 + 拡張子「.avi」")
		# packでウィジェット配置、padyで垂直方向の空白(単位はpixel)
        self.start_button.pack(pady=5)
        self.stop_button.pack(pady=5)
        self.label.pack(pady=10)

	# ラベル表示を更新する
    def update_label(self):
    	# 録画停止フラッグに応じてラベルのtextを変える
        if self.config.stop_flag == 0:
            self.label.config(text='レコーディング中')
        elif self.config.stop_flag == 1:
            self.label.config(text='ファイル名の指定 + 拡張子「.avi」')

	# 録画スタート
    def record_start(self):
    	# 録画開始フラッグを1にする
        self.config.start_flag = 1
        # カメラを初期化
        self.config.cap = cv2.VideoCapture(0)
        # record_controlメソッドを新しいスレッド実行
        self.record_thread = threading.Thread(target=self.record_control)
        self.record_thread.start()

		# ラベルの更新
        self.update_label()

	# 録画ストップ
    def record_stop(self):
    	# start_flagが1の場合
        if self.config.start_flag == 1:
        	# stop_flagを1にする
            self.config.stop_flag = 1
            # スレッド処理終了
            self.record_thread.join()
            # カメラキャプチャを解放
            self.config.cap.release()
            # 画面録画を保存
            self.record_save()

			# ラベルの更新
            self.update_label()

			# stop_flagを0にする
            self.config.stop_flag = 0
            # start_flagを0にする
            self.config.start_flag = 0
        else:
            print('ファイル名の指定 + 拡張子「.avi」')

	# 録画映像の保存
    def record_save(self):
    	# ファイル名入力フィールドからファイル名を取得
        filename = self.filename_entry.get()
        if not filename:
        	# ファイル名が入力されていない場合、ダイアログでファイル名を尋ねる
            filename = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI files", "*.avi")])
            if not filename:
                return

		# 録画したフレームを動画として保存
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(filename, fourcc, self.config.fps, (self.config.w, self.config.h))
        for frame in self.config.img_list:
            video.write(frame)
        video.release()

	# recordThreadの制御を行う
    def record_control(self):
    	# RecordThreadのインスタンスを生成し、そのloopメソッドを実行
        rec = RecordThread(self.config)
        rec.loop()

def main():
    app = CaptureApp()
    app.title("Camera Capture App")
    app.geometry("300x200")
    app.mainloop()

if __name__ == '__main__':
    main()
