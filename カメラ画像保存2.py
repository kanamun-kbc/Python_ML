import cv2
import datetime

# カメラのデバイス番号。通常は0か1です。
camera_index = 0
# VideoCaptureオブジェクトを取得
cap = cv2.VideoCapture(camera_index)
# カメラが正常に開かれたかどうかを確認
if not cap.isOpened():
	print("Error: Could not open camera.")
	exit()
# ウィンドウを作成
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

while True:
 	# カメラからフレームを読み込む
	ret, frame = cap.read()
 	# フレームが正常に読み込まれたかどうかを確認
	if not ret:
		print("Error: Could not read frame.")
		break
 	# フレームを表示
	cv2.imshow("Camera Feed", frame)
 	# 'q' キーが押されたらループを終了
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
 	# 'c' キーが押されたら静止画を保存
	if cv2.waitKey(10) & 0xFF == ord('c'):
  	# 静止画保存用の設定
		output_file = f"captured_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
  		# フレームを保存する
		cv2.imwrite(output_file, frame)
		print("画像を保存しました。")


# リソースを解放
cap.release()
cv2.destroyAllWindows()