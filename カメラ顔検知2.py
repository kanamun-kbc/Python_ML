import cv2
import face_recognition

# 自分の顔の特徴を読み込むための空のリスト
my_face_encodings = []

# 自分の顔写真
my_images = ["My_Face_00.jpg", "My_Face_01.jpg", "My_Face_02.jpg", "My_Face_03.jpg", "My_Face_04.jpg", "My_Face_05.jpg", "My_Face_06.jpg", "My_Face_07.jpg", "My_Face_08.jpg", "My_Face_09.jpg", "My_Face_10.jpg"]
for image in my_images:
    # 顔写真を読み込む
    img = face_recognition.load_image_file(image)
    # 顔の特徴を抽出し、リストに追加
    my_face_encodings.append(face_recognition.face_encodings(img)[0])

# カメラのセットアップ
camera = cv2.VideoCapture(0)

while True:
    # カメラから映像を取得
    ret, frame = camera.read()
    if not ret:
        break

    # 現在のフレームから顔の位置を検出
    face_locations = face_recognition.face_locations(frame)
    # 検出された顔の特徴を抽出
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # 認識された顔をチェック
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 保存されている顔の特徴を比較し、一致するかをチェック
        matches = face_recognition.compare_faces(my_face_encodings, face_encoding)
        # 一致する顔があったら
        if True in matches:
            # 四角で囲む
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # 映像を表示
    cv2.imshow("Camera", frame)

    # 'q'を押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラの解放
camera.release()
# 全てのOpenCVウィンドウを閉じる
cv2.destroyAllWindows()
