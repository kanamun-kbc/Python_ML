import cv2
import datetime

# カメラのデバイス番号。通常は0か1です。
camera_index = 0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# 動画の保存先とファイル形式を指定
output_file = "captured_video.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
fps = 30
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (640, 480))

# 開始時刻を取得
start_time = datetime.datetime.now()

while True:
    # カメラからフレームを読み込む
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # フレームを動画に書き込む
    video_writer.write(frame)

    # カメラ映像を表示（オプション）
    cv2.imshow("Camera Feed", frame)

    # 5秒経過またはESCキーが押されたらループを終了
    elapsed_time = datetime.datetime.now() - start_time
    if elapsed_time.total_seconds() >= 5 or cv2.waitKey(1) == 27:
        break

# リソースを解放
cap.release()
video_writer.release()
cv2.destroyAllWindows()
