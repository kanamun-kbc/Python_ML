import cv2

# 顔検出のためのCascade分類器の読み込み
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# カメラキャプチャオブジェクトの初期化
cap = cv2.VideoCapture(0)

while True:
    # カメラからフレームを読み込む
    ret, frame = cap.read()
    if not ret:
        break

    # グレースケールに変換（顔検出の効率化）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔の検出
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 検出された顔を四角で囲む
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 結果の表示
    cv2.imshow('Face Detection', frame)

    # 'q'キーが押されたらループから抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャの解放とウィンドウの破棄
cap.release()
cv2.destroyAllWindows()
